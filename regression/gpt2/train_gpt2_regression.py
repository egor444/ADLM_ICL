import os
import re
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_extraction.data_manager import DataManager
from feature_engineering.featureselection import FeatureSelector
from matplotlib import pyplot as plt
import logging

# Configuration
config = {
    "TASK": "regression",
    "N_FEATURES": 200,
    "N_NEIGHBORS": 40,
    "VARIANCE_THRESHOLDS": {
        'radiomics_shape': 1e-6,
        'radiomics_texture': 1e-5,
        'radiomics_intensity': 1e-5,
        'embeddings': 1e-7,
        'tabular': 1e-6,
        'default': 1e-6
    },
    "CORRELATION_THRESHOLDS": {
        'radiomics_shape': 0.85,
        'radiomics_texture': 0.92,
        'radiomics_intensity': 0.88,
        'embeddings': 0.95,
        'tabular': 0.90,
        'default': 0.90
    },
    "RANDOM_STATE": 42,
    "FEATURE_SELECTION_STRATEGIES": {
        "multi_method_scoring": False,
        "hierarchical_filtering": True,
        "group_based_selection": False
    },
}

class GPT2Regressor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()
        self.device = device

    def predict(self, prompts, num_return_sequences=3):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=900).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        grouped = [decoded[i*num_return_sequences:(i+1)*num_return_sequences] for i in range(len(prompts))]

        preds = []
        for group in grouped:
            nums = []
            for txt in group:
                match = re.search(r"(\d{2,3}\.\d+)", txt)  
                if match:
                    nums.append(float(match.group(1)))
            pred = np.median(nums) if nums else 0.0
            preds.append(pred)
        return np.array(preds)

def test_gpt2_regression(dm):
    train_set = dm.get_train()
    test_set = dm.get_test()

    X_train = train_set.drop(columns=['age'])
    y_train = train_set['age']
    X_test = test_set.drop(columns=['age'])
    y_test = test_set['age']

    fs = FeatureSelector(config, verbose=False)
    X_train = fs.fit_transform(X_train, y_train)
    X_test = fs.transform(X_test)

    logging.info(f"FS and scaling complete, training set size: {len(X_train)}, Test set size: {len(X_test)}")

    scaler_X = StandardScaler().fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    knn = NearestNeighbors(n_neighbors=config["N_NEIGHBORS"])
    knn.fit(X_train)

    model = GPT2Regressor()
    preds, maes = [], []

    for i in range(len(X_test)):
        x_query = X_test[i]
        y_true = y_test.values[i]
        indices = knn.kneighbors([x_query], return_distance=False)[0]
        X_context = X_train[indices]
        y_context = y_train.values[indices]

        prompt = "Predict the age from input features.\n"
        for x, y_val in zip(X_context, y_context):
            x_str = ", ".join([f"{val:.2f}" for val in x[:20]])  
            prompt += f"Input: {x_str}\nOutput: {y_val:.1f}\n"
            
        x_query_str = ", ".join([f"{val:.2f}" for val in x_query[:20]])
        prompt += f"Input: {x_query_str}\nOutput:"


        y_pred = model.predict([prompt])[0]
        mae = mean_absolute_error([y_true], [y_pred])

        preds.append(y_pred)
        maes.append(mae)

        logging.info(f"Sample {i+1}/{len(X_test)}: Pred={y_pred:.2f}, True={y_true:.2f}, MAE={mae:.2f}")

    preds = np.array(preds)
    maes = np.array(maes)
    overall_mae = mean_absolute_error(y_test, preds)
    logging.info(f"Overall MAE: {overall_mae:.2f}")

    # Save results
    os.makedirs("results/gpt2_results/test_set", exist_ok=True)
    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": preds,
        "MAE": maes
    })
    results_df.to_csv("results/gpt2_results/test_set/gpt2_regression_results.csv", index=False)

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('GPT-2 Regression Predictions vs Actual')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.grid()
    plt.savefig('results/gpt2_results/gpt2_regression_predictions.png')
    plt.close()

    # MAE per sample
    plt.figure(figsize=(10, 6))
    plt.plot(maes, marker='o', linestyle='-', color='blue')
    plt.title('Mean Absolute Error per Test Sample (GPT-2)')
    plt.xlabel('Test Sample Index')
    plt.ylabel('MAE')
    plt.grid()
    plt.savefig('results/gpt2_results/gpt2_regression_mae.png')
    plt.close()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='outfiles6/reg2.log', filemode='w')
    logging.info("Testing GPT-2 with ICL regression ...")
    dm_logger = logging.getLogger('DATAMANAGER')
    dm = DataManager("regression", "emb", "rfat", "verbose", logger=dm_logger)
    test_gpt2_regression(dm)
    logging.info("Test completed.")

if __name__ == "__main__":
    main()
