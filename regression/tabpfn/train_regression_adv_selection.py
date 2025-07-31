import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_extraction.data_manager import DataManager
from tabpfn import TabPFNRegressor
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

class ContextSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_query, x_train):
        q_embed = self.query_encoder(x_query).expand(x_train.size(0), -1)
        x_embed = self.item_encoder(x_train)
        pair_embed = torch.cat([q_embed, x_embed], dim=1)
        logits = self.scorer(pair_embed).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs


class TabPFNRegressorRow:
    def __init__(self, device='cpu'):
        self.device = device

    def fit_predict(self, X_train, y_train, X_test):
        tabpfn = TabPFNRegressor(device=self.device)
        tabpfn.fit(X_train, y_train)
        return tabpfn.predict(X_test)


# ---- REINFORCE loss ----
def reinforce_loss(log_probs, preds, y_true):
    reward = -((preds - y_true) ** 2)  # negative MSE = reward
    return -(log_probs + reward).mean()

# ---- Parallelized training loop ----
def train_regression_model():
    N_EPOCHS = 10
    LR = 1e-3
    HIDDEN_DIM = 64
    CPU_COUNT = 16

    dm_logger = logging.getLogger('DATAMANAGER')
    dm = DataManager("regression", "emb", "verbose", "pca", logger=dm_logger)

    fold_count = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    for fold in range(fold_count):
        test_set = dm.get_fold_data_set([fold])
        X_test = test_set.drop(columns=['age']).values
        y_test = test_set['age'].values
        input_dim = X_test.shape[1]

        context_selector = ContextSelector(input_dim, HIDDEN_DIM).to(device)
        optimizer = optim.Adam(context_selector.parameters(), lr=LR)
        context_selector.train()

        rest_folds = [j for j in range(fold_count) if j != fold]
        epoch_losses = [[] for _ in range(N_EPOCHS)]
        for epoch in range(N_EPOCHS):
            for val_fold in rest_folds:
                train_set = dm.get_fold_data_set([f for f in rest_folds if f != val_fold])
                val_set = dm.get_fold_data_set([val_fold])

                X_train = train_set.drop(columns=['age']).values
                y_train = train_set['age'].values
                X_val = val_set.drop(columns=['age']).values
                y_val = val_set['age'].values

                X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

                # ---- Parallel prediction function ----
                def predict_for_query(i):
                    x_query = torch.tensor(X_val[i], dtype=torch.float32, device=device)
                    probs = context_selector(x_query, X_train_tensor)
                    bernoulli_dist = torch.distributions.Bernoulli(probs)
                    actions = bernoulli_dist.sample()
                    log_prob = bernoulli_dist.log_prob(actions).sum()

                    actions_np = actions.detach().cpu().numpy()
                    selected_idx = np.where(actions_np > 0.5)[0]
                    if len(selected_idx) < 2:  # enforce minimum size
                        selected_idx = np.arange(len(X_train))

                    X_selected = X_train[selected_idx]
                    y_selected = y_train[selected_idx]

                    tabpfn_row = TabPFNRegressorRow(device=device)
                    pred = tabpfn_row.fit_predict(X_selected, y_selected, X_val[i].reshape(1, -1))
                    return pred[0], log_prob
                logging.info(f"Fold {fold}, Epoch {epoch}, Val Fold {val_fold}: Starting predictions...")
                results = Parallel(n_jobs=CPU_COUNT)(
                    delayed(predict_for_query)(i) for i in range(len(X_val))
                )
                logging.info(f"Fold {fold}, Epoch {epoch}, Val Fold {val_fold}: Predictions completed.")
                preds, log_probs = zip(*results)
                preds_tensor = torch.tensor(preds, dtype=torch.float32, device=device)
                log_probs_tensor = torch.stack(log_probs)

                y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)

                # ---- REINFORCE loss ----
                loss = reinforce_loss(log_probs_tensor, preds_tensor, y_val_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses[epoch].append(loss.item())
                logging.info(f"Fold {fold}, Epoch {epoch}, Val Fold {val_fold}, Loss: {loss.item()}, MAE: {mean_absolute_error(y_val, preds)}")
                # save preds and true values for later analysis
                pd.DataFrame({
                    'preds': preds,
                    'true': y_val
                }).to_csv(f'outfiles2/foldpreds/regression_fold_{fold}_epoch_{epoch}_val_fold_{val_fold}.csv', index=False)
        
        logging.info(f"Fold {fold} training completed.")

        # ---- Plot training loss ----
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(N_EPOCHS), [np.mean(epoch_loss) for epoch_loss in epoch_losses], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold} Training Loss')
        plt.legend()
        plt.savefig(f'outfiles2/regression_fold_{fold}_loss.png')
        plt.close()

        # ---- Evaluate on test fold (deterministic) ----
        context_selector.eval()
        preds_test = []
        X_train_all = dm.get_fold_data_set(rest_folds).drop(columns=['age']).values
        y_train_all = dm.get_fold_data_set(rest_folds)['age'].values
        X_train_tensor_all = torch.tensor(X_train_all, dtype=torch.float32, device=device)

        for i in range(len(X_test)):
            x_query = torch.tensor(X_test[i], dtype=torch.float32, device=device)
            with torch.no_grad():
                probs = context_selector(x_query, X_train_tensor_all)
                topk_idx = torch.topk(probs, max(2, int(0.1 * len(X_train_all)))).indices
                X_selected = X_train_all[topk_idx.cpu().numpy()]
                y_selected = y_train_all[topk_idx.cpu().numpy()]

                tabpfn_row = TabPFNRegressorRow(device=device)
                pred = tabpfn_row.fit_predict(X_selected, y_selected, X_test[i].reshape(1, -1))
                preds_test.append(pred[0])

        mae_test = mean_absolute_error(y_test, preds_test)
        mse_test = mean_squared_error(y_test, preds_test)
        r2_test = r2_score(y_test, preds_test)
        logging.info(f"Fold {fold} Test MSE: {mse_test}")
        logging.info(f"Fold {fold} Test R2: {r2_test}")
        logging.info(f"Fold {fold} Test MAE: {mae_test}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='outfiles2/reg2.log', filemode='w')
    logging.info("Testing TabPFN content selection...")
    train_regression_model()
    logging.info("Regression model training completed.")