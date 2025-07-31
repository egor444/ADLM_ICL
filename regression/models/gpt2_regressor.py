import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import Counter
import re


AGE_BINS = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]


def bin_label(age):
    for lo, hi in AGE_BINS:
        if lo <= age < hi:
            return f"between {lo} and {hi}"
    return "unknown"


def bin_midpoint(label):
    match = re.match(r"between (\d+) and (\d+)", label)
    if match:
        return (int(match.group(1)) + int(match.group(2))) / 2
    return 0.0


class InContextRegressionDataset(Dataset):
    def __init__(self, X, y, X_train, y_train, region_bins=(3, 20)):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.region_bins = region_bins

        self.nn_orig = NearestNeighbors(n_neighbors=50).fit(X_train)

        hist, bin_edges = np.histogram(y_train, bins=10)
        bin_ids = np.digitize(y_train, bin_edges[:-1], right=True)
        inv_freq = {i: 1 / (count + 1e-6) for i, count in enumerate(hist)}
        weights = np.array([
            inv_freq.get(max(0, min(bin_id - 1, len(inv_freq) - 1)), 1.0)
            for bin_id in bin_ids
        ])
        probs = weights / weights.sum()

        aug_indices = np.random.choice(len(y_train), size=int(0.5 * len(y_train)), p=probs)
        self.X_aug = X_train[aug_indices]
        self.y_aug = y_train[aug_indices]
        self.nn_aug = NearestNeighbors(n_neighbors=50).fit(self.X_aug)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        query = self.X[idx]
        target = self.y[idx]

        distances, indices = self.nn_orig.kneighbors([query], n_neighbors=50)
        neighbors_y = [self.y_train[i] for i in indices[0]]
        close_count = sum(abs(np.array(neighbors_y) - target) < 0.2)

        if close_count <= self.region_bins[0]:
            region = 'few-shot'
            k_context = 2
            aug_k = 2
        elif close_count <= self.region_bins[1]:
            region = 'medium-shot'
            k_context = 4
            aug_k = 4
        else:
            region = 'many-shot'
            k_context = 6
            aug_k = 6

        indices_orig = self.nn_orig.kneighbors([query], n_neighbors=k_context + 1)[1][0][1:]
        indices_aug = self.nn_aug.kneighbors([query], n_neighbors=aug_k)[1][0]

        X_neighbors = np.vstack([self.X_train[indices_orig], self.X_aug[indices_aug]])
        y_neighbors = np.concatenate([self.y_train[indices_orig], self.y_aug[indices_aug]])

        prompt = "The age is one of these ranges: 40-50, 50-60, ..., 80-90 years.\n"
        for x, y_val in zip(X_neighbors, y_neighbors):
            x_str = ", ".join([f"{val:.1f}" for val in x[:20]])
            age_text = bin_label(y_val)
            prompt += f"The patient has features: {x_str}. The age is {age_text}.\n"

        x_query = ", ".join([f"{val:.1f}" for val in query[:20]])
        prompt += f"The patient has features: {x_query}. The age is"

        return prompt, target, region


class GPT2Regressor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()
        self.device = device

    def predict(self, prompts, num_return_sequences=5):
        trunc_prompts = [p[-1000:] for p in prompts]
        inputs = self.tokenizer(trunc_prompts, return_tensors="pt", padding=True, truncation=True, max_length=900).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                top_k=50,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        grouped = [decoded[i*num_return_sequences:(i+1)*num_return_sequences] for i in range(len(prompts))]

        preds = []
        for group in grouped:
            bin_preds = []
            for txt in group:
                match = re.search(r"between (\d+) and (\d+)", txt)
                if match:
                    mid = (int(match.group(1)) + int(match.group(2))) / 2
                    bin_preds.append(mid)
            pred = np.median(bin_preds) if bin_preds else 0.0
            preds.append(pred)
        return np.array(preds)


def run_adaptive_icl_regression():
    print("Running adaptive in-context learning for regression...")

    train_path = "../data/regression/ALL_pca_train.csv"
    test_path = "../data/regression/ALL_pca_test.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=['age']).values
    y_train = df_train['age'].values
    X_test = df_test.drop(columns=['age']).values
    y_test = df_test['age'].values

    scaler_X = StandardScaler().fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    test_ds = InContextRegressionDataset(X_test, y_test, X_train, y_train)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    model = GPT2Regressor()

    y_preds, y_trues, regions = [], [], []
    for batch in test_loader:
        prompts, targets, batch_regions = batch
        preds = model.predict(prompts)
        y_preds.append(preds)
        y_trues.append(targets.numpy())
        regions.extend(batch_regions)

    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)

    mse = mean_squared_error(y_trues, y_preds)
    mae = mean_absolute_error(y_trues, y_preds)
    r2 = r2_score(y_trues, y_preds)
    print(f"\nAdaptive ICL GPT2 Regression Results:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²:  {r2:.2f}")

    print("\nPerformance by Region:")
    for region_type in ['few-shot', 'medium-shot', 'many-shot']:
        idxs = [i for i, r in enumerate(regions) if r == region_type]
        if not idxs:
            continue
        region_y = y_trues[idxs]
        region_pred = y_preds[idxs]
        print(f"{region_type:12}: MAE={mean_absolute_error(region_y, region_pred):.2f}, MSE={mean_squared_error(region_y, region_pred):.2f}, R2={r2_score(region_y, region_pred):.2f} (n={len(idxs)})")

if __name__ == "__main__":
    run_adaptive_icl_regression()
