import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter
import random
from tqdm import tqdm


class InContextClassificationDataset(Dataset):
    def __init__(self, X, y, X_train, y_train, max_features=30, aug_factor=0.5, template='natural', region_bins=(3, 20)):
        self.X = X
        self.y = y
        self.template = template
        self.max_features = max_features
        self.region_bins = region_bins

        self.X_train = X_train
        self.y_train = y_train

        self.nn_orig = NearestNeighbors(n_neighbors=50).fit(X_train)

        class_counts = Counter(y_train)
        total = sum(class_counts.values())
        inv_freq = {cls: total / (count + 1e-6) for cls, count in class_counts.items()}
        weights = np.array([inv_freq[label] for label in y_train])
        probs = weights / weights.sum()

        aug_indices = np.random.choice(len(y_train), size=int(aug_factor * len(y_train)), p=probs)
        self.X_aug = X_train[aug_indices]
        self.y_aug = y_train[aug_indices]
        self.nn_aug = NearestNeighbors(n_neighbors=50).fit(self.X_aug)

    def get_prompt(self, X_neighbors, y_neighbors, x_query):
        prompt = ""
        for x, y in zip(X_neighbors, y_neighbors):
            x_str = ", ".join([f"{val:.2f}" for val in x[:self.max_features]])
            label_text = "positive" if y == 1 else "negative"
            if self.template == "natural":
                prompt += f"The patient has features: {x_str}. The diagnosis is {label_text}.\n"
            elif self.template == "simple":
                prompt += f"{x_str} => {label_text}\n"
            else:
                prompt += f"Features: {x_str}. Label: {label_text}.\n"

        query_str = ", ".join([f"{val:.2f}" for val in x_query[:self.max_features]])
        if self.template == "natural":
            prompt += f"The patient has features: {query_str}. The diagnosis is"
        elif self.template == "simple":
            prompt += f"{query_str} =>"
        else:
            prompt += f"Features: {query_str}. Label:"
        return prompt

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        query = self.X[idx]
        label = self.y[idx]

        distances, indices = self.nn_orig.kneighbors([query], n_neighbors=50)
        neighbor_labels = [self.y_train[i] for i in indices[0]]
        same_label_count = neighbor_labels.count(label)

        if same_label_count <= self.region_bins[0]:
            region = 'few-shot'
            k_context = 5
            aug_k = 5
        elif same_label_count <= self.region_bins[1]:
            region = 'medium-shot'
            k_context = 3
            aug_k = 4
        else:
            region = 'many-shot'
            k_context = 1
            aug_k = 1

        indices_orig = self.nn_orig.kneighbors([query], n_neighbors=k_context + 1)[1][0][1:]
        indices_aug = self.nn_aug.kneighbors([query], n_neighbors=aug_k)[1][0]

        X_neighbors = np.vstack([self.X_train[indices_orig], self.X_aug[indices_aug]])
        y_neighbors = np.concatenate([self.y_train[indices_orig], self.y_aug[indices_aug]])

        prompt = self.get_prompt(X_neighbors, y_neighbors, query)
        return prompt, label, region


class GPT2ICLLabelDecoder:
    def __init__(self, label_texts=["negative", "positive"], device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.model.eval()
        self.label_texts = label_texts
        self.label_tokens = [self.tokenizer.encode(label, add_special_tokens=False)[0] for label in label_texts]

    def predict(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            last_token_idxs = inputs['attention_mask'].sum(dim=1) - 1
            next_token_logits = logits[range(len(prompts)), last_token_idxs]
            probs = torch.softmax(next_token_logits[:, self.label_tokens], dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
        return preds, probs.cpu().numpy()


def run_icl_with_region_analysis(template='natural'):
    print(f"\nRunning Adaptive-k ICL with template='{template}'...\n")

    df_train = pd.read_csv("../data/classification/cancer/ALL_pca_train.csv").drop(columns=["eid"])
    df_test = pd.read_csv("../data/classification/cancer/ALL_pca_test.csv").drop(columns=["eid"])

    X_train = df_train.drop(columns=['target']).values
    y_train = df_train['target'].values
    X_test = df_test.drop(columns=['target']).values
    y_test = df_test['target'].values

    selector = SelectKBest(score_func=f_classif, k=30)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ds = InContextClassificationDataset(X_test, y_test_enc, X_train, y_train_enc, template=template)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    model = GPT2ICLLabelDecoder(label_texts=["negative", "positive"])

    all_preds, all_trues, all_regions = [], [], []

    for batch in tqdm(loader):
        prompts, targets, regions = batch
        preds, _ = model.predict(prompts)
        all_preds.extend(preds)
        all_trues.extend(targets.numpy())
        all_regions.extend(regions)

    y_pred = np.array(all_preds)
    y_true = np.array(all_trues)

    print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_.astype(str)))

    print("\nDetailed Accuracy by Region and Class:")
    regions_set = ["few-shot", "medium-shot", "many-shot"]
    classes = [0, 1]

    for region in regions_set:
        idxs = [i for i, r in enumerate(all_regions) if r == region]
        if not idxs:
            continue
        y_region = y_true[idxs]
        y_pred_region = y_pred[idxs]
        print(f"\n{region.upper()} (n={len(idxs)}):")
        for cls in classes:
            cls_idxs = [i for i in range(len(y_region)) if y_region[i] == cls]
            if cls_idxs:
                acc = accuracy_score(y_region[cls_idxs], y_pred_region[cls_idxs])
                print(f"  Class {cls}: {acc:.2f} ({len(cls_idxs)} samples)")


if __name__ == "__main__":
    run_icl_with_region_analysis(template="natural")