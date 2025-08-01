from tabpfn import TabPFNRegressor, TabPFNClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import GridSearchCV

import time


def get_pca_data(data_df, n_components=0.95):
    """Apply PCA to the data and return the transformed data."""
    print("\t\tApplying PCA to the data...")
    age = data_df['age'].values
    eids = data_df['eid'].values
    data = data_df.drop(columns=['eid','age']) 

    pca = PCA(n_components=n_components)  
    print(f"\t\tPCA fitting..")
    pca.fit(data)
    pca_data = pca.transform(data)
    pca_data = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    pca_data['age'] = age
    pca_data['eid'] = eids
    print(f"\t\tPCA applied, total features: {pca_data.shape[1]-2}")
    return pca_data, pca

def fit_model(model, data_path, target_col='age',use_pca=False):
    """Fitting the TabPFN model to the data."""
    print("Begginging model fit\n\tLoading data..")
    data = pd.read_csv(data_path)
    if use_pca:
        data, pca = get_pca_data(data, n_components=0.95)
    print("\tData loaded, shape:", data.shape)
    cols = data.columns.tolist()
    cols.remove(target_col)

    X = data[cols].values
    y = data[target_col].values
    print(f"\tPreprocessing done. X shape: {X.shape}, y shape: {y.shape}")
    print("\tFitting model..")
    model.fit(X, y)
    print("\tModel fitting complete.")
    return model, pca if use_pca else None

def test_model(model, data_path, target_col='age', use_pca=None):
    """Testing the TabPFN model on the data."""
    filename = data_path.split("/")[-1]
    print(f"Beginning model test. Loading test data from {filename}..")
    test_data = pd.read_csv(data_path)
    y_test = test_data[target_col].values
    if use_pca:
        test_data = use_pca.transform(test_data.drop(columns=[target_col]))
    else:
        test_data = test_data.drop(columns=[target_col]).values
    X_test = test_data
    print(f"\tTest data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

    print("\tMaking predictions..")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test complete; MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")
    return y_pred, y_test, mse, mae, r2

def plot_predictions(y_test, y_pred, mse, mae, r2, test_name):
    """Plotting predictions vs true values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'TabPFN Predictions\nMAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}')
    plt.savefig(f'outputs/tabpfn_{test_name}.png')
    plt.show()
    

def run_regression():
    print("Running TabPFN")
    data_path = "../data/regression/ALL_pca_train.csv"
    test_path = "../data/regression/ALL_pca_test.csv"
    USE_PCA = False
    model = TabPFNRegressor(device='cuda', ignore_pretraining_limits=True)
    
    print("\tFitting model..")
    train_time = time.time()
    model, pca = fit_model(model, data_path, use_pca=USE_PCA)
    train_time = time.time() - train_time
    print(f"\tModel fitted in {train_time:.2f} seconds.")

    test_time_1 = time.time()
    ypred, ytest, mse, mae, r2 = test_model(model, test_path, use_pca=pca)
    test_time_1 = time.time() - test_time_1
    print(f"\tModel tested on ALL data in {test_time_1:.2f} seconds.")
    plot_predictions(ytest, ypred, mse, mae, r2, "ALL_regression")
    print("TabPFN regression test finished.")


def run_test_classification():
    print("Running TabPFN classification test; loading data..")
    data_path_train = "../data/classification/ALL_pca_train.csv"
    data_train = pd.read_csv(data_path_train)

    # balance classes 
    #print("\tBalancing classes..")
    #class_counts = data_train['target'].value_counts()
    #min_class_count = class_counts.min()
    #balanced_data = pd.concat([
    #    data_train[data_train['target'] == cls].sample(min_class_count, random_state=42)
    #    for cls in class_counts.index
    #])
    #data_train = balanced_data.reset_index(drop=True)

    X = data_train.drop(columns=['target']).values
    y = data_train['target'].values

    model = TabPFNClassifier(device='cuda', ignore_pretraining_limits=True)
    print("\tFitting model..")
    train_time = time.time()
    model.fit(X, y)
    train_time = time.time() - train_time
    print(f"\tModel fitted in {train_time:.2f} seconds.")
    print("\tModel fitting complete. Testing model..")
    data_path_test = "../data/classification/ALL_pca_test.csv"
    data_test = pd.read_csv(data_path_test)

    
    print(f"\tTest data shape: {data_test.shape}")

    X_test = data_test.drop(columns=['target']).values
    y_test = data_test['target'].values
    test_time = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - test_time
    print(f"\tModel testing complete in {test_time:.2f} seconds. Calculating metrics..")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'TabPFN Classification Confusion Matrix\nAccuracy: {acc:.2f}')
    plt.savefig('outputs/tabpfn_classification_ALL_cm.png')
    plt.show()

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    # plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for TabPFN')
    plt.legend(loc='lower right')
    plt.savefig('outputs/tabpfn_classification_ALL_roc.png')
    plt.show()
    print(f"TabPFN classification test finished. Accuracy: {acc:.2f}, ROC AUC: {roc_auc:.2f}")




def run_tabpfn_classification_tuning():
    print("Running TabPFN classification tuning; loading data..")
    data_path_train = "../data/classification/cancer/ALL_pca_train.csv"
    data_train = pd.read_csv(data_path_train)

    X = data_train.drop(columns=['target']).values
    y = data_train['target'].values

    
    param_grid = {
    'n_estimators': [1, 4, 8, 16, 32],
    'device': ['cuda'],
    'ignore_pretraining_limits': [True],
    'softmax_temperature': [0.5, 1.0, 1.5, 2.0, 3.0]
}

    
    model = TabPFNClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2)
    print("\tFitting model with grid search..")
    train_time = time.time()
    grid_search.fit(X, y)
    train_time = time.time() - train_time
    print(f"\tBest params: {grid_search.best_params_}")
    print(f"\tBest cross-validated accuracy: {grid_search.best_score_:.2f}")
    print(f"\tGrid search completed in {train_time:.2f} seconds.")

    
    best_model = grid_search.best_estimator_
    data_path_test = "../data/classification/cancer/ALL_pca_test.csv"
    data_test = pd.read_csv(data_path_test)
    print(f"\tTest data shape: {data_test.shape}")

    X_test = data_test.drop(columns=['target']).values
    y_test = data_test['target'].values
    test_time = time.time()
    y_pred = best_model.predict(X_test)
    test_time = time.time() - test_time
    print(f"\tModel testing complete in {test_time:.2f} seconds. Calculating metrics..")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'TabPFN classification tuning Confusion Matrix\nAccuracy: {acc:.2f}')
    plt.savefig('outputs/tabpfn_classification_tuning_ALL_cm.png')
    plt.show()

    y_prob = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    # plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for TabPFN')
    plt.legend(loc='lower right')
    plt.savefig('outputs/tabpfn_classification_tuning_ALL_roc.png')
    plt.show()
    print(f"TabPFN classification tuning test finished. Accuracy: {acc:.2f}, ROC AUC: {roc_auc:.2f}")



from tabpfn import TabPFNRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_tabpfn_regression_tuning():
    print("Running TabPFN regression tuning; loading data..")
    data_path_train = "../data/regression/ALL_pca_train.csv"
    data_train = pd.read_csv(data_path_train)

    X = data_train.drop(columns=['age']).values  # Use your regression target column
    y = data_train['age'].values

    param_grid = {
        'n_estimators': [1, 4, 8, 16],
        'device': ['cuda'],
        'ignore_pretraining_limits': [True],
        'softmax_temperature': [0.5, 1.0, 2.0]
    }

    model = TabPFNRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
    print("\tFitting model with grid search..")
    train_time = time.time()
    grid_search.fit(X, y)
    train_time = time.time() - train_time
    print(f"\tBest params: {grid_search.best_params_}")
    print(f"\tBest cross-validated MSE: {-grid_search.best_score_:.2f}")
    print(f"\tGrid search completed in {train_time:.2f} seconds.")

    best_model = grid_search.best_estimator_
    data_path_test = "../data/regression/ALL_pca_test.csv"
    data_test = pd.read_csv(data_path_test)
    X_test = data_test.drop(columns=['age']).values
    y_test = data_test['age'].values

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test set results - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

# ===================== GPT2 Tabular Regression =====================

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from sklearn.base import BaseEstimator, RegressorMixin

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=4, n_head=4):
        super().__init__()
        config = GPT2Config(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=n_positions,
            n_ctx=n_positions,
            n_inner=None,
            vocab_size=1,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.embed = nn.Linear(n_dims + 1, n_embd)
        self.transformer = GPT2Model(config)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        B, N, D = xs_b.shape
        ys_b = ys_b[..., None]
        seq = torch.cat([xs_b, ys_b], dim=-1)
        return seq

    def forward(self, xs, ys):
        seq = self._combine(xs, ys)
        emb = self.embed(seq)
        out = self.transformer(inputs_embeds=emb).last_hidden_state
        preds = self._read_out(out).squeeze(-1)
        return preds

class GPT2TabularRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=4, n_head=4, epochs=3, lr=1e-4, device='cuda'):
        self.n_dims = n_dims
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
            y = y.reshape(1, y.shape[0])
        self.model_ = TransformerModel(
            n_dims=self.n_dims,
            n_positions=self.n_positions,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        self.model_.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            xs = torch.tensor(X, dtype=torch.float32, device=self.device)
            ys = torch.tensor(y, dtype=torch.float32, device=self.device)
            preds = self.model_(xs, ys)
            loss = loss_fn(preds, ys)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss.item():.4f}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        self.model_.eval()
        with torch.no_grad():
            xs = torch.tensor(X, dtype=torch.float32, device=self.device)
            ys = torch.zeros(xs.shape[:-1], dtype=torch.float32, device=self.device)
            preds = self.model_(xs, ys)
        return preds.cpu().numpy().flatten()

def run_gpt2_tabular_regression():
    print("Running GPT2 tabular regression ...")
    train_path = "../data/regression/ALL_pca_train.csv"
    test_path = "../data/regression/ALL_pca_test.csv"

    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=['age']).values
    y_train = df_train['age'].values
    X_test = df_test.drop(columns=['age']).values
    y_test = df_test['age'].values

    n_dims = X_train.shape[1]
    n_positions = X_train.shape[0]

    model = GPT2TabularRegressor(
        n_dims=n_dims,
        n_positions=n_positions,
        epochs=3,
        lr=1e-4,
        device='cuda'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"GPT2 Regression Results - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")


from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleTransformerModel(nn.Module):
    def __init__(self, n_dims, n_embd=32, n_layer=2, n_head=2):
        super().__init__()
        config = GPT2Config(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=1,
            n_ctx=1,
            vocab_size=1,
        )
        self.embed = nn.Linear(n_dims, n_embd)
        self.transformer = GPT2Model(config)
        self.read_out = nn.Linear(n_embd, 1)
    def forward(self, x):
        # x: [B, D]
        emb = self.embed(x).unsqueeze(1)  # [B, 1, n_embd]
        out = self.transformer(inputs_embeds=emb).last_hidden_state  # [B, 1, n_embd]
        pred = self.read_out(out).squeeze(1)  # [B, 1] -> [B]
        return pred

def run_gpt2_tabular_regression_rowwise():
    print("Running improved GPT2 tabular regression ...")
    train_path = "../data/regression/ALL_pca_train.csv"
    test_path = "../data/regression/ALL_pca_test.csv"
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Load and scale data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.drop(columns=['age']).values
    y_train = df_train['age'].values
    X_test = df_test.drop(columns=['age']).values
    y_test = df_test['age'].values

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Dataset and loader
    train_ds = TabularDataset(X_train, y_train_scaled)
    test_ds = TabularDataset(X_test, y_test_scaled)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleTransformerModel(n_dims=X_train.shape[1], n_embd=32, n_layer=2, n_head=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training
    model.train()
    for epoch in range(20):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/20 - Train Loss: {total_loss/len(train_ds):.4f}")

    # Evaluation
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            pred = model(xb).squeeze().cpu().numpy()
            preds.append(pred)
    y_pred_scaled = np.concatenate(preds)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Improved GPT2 Regression Results - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import GPT2Model, GPT2Config
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

class IMContextTabularDataset(Dataset):
    """
    Each item is a (context_X, context_y, target_X, target_y) tuple.
    For simplicity, here we use all but one row as context, and one as target.
    """
    def __init__(self, X, y, context_size=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.N = X.shape[0]
        self.context_size = context_size or (self.N - 1)
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        # Use all but idx as context, idx as target
        context_mask = np.ones(self.N, dtype=bool)
        context_mask[idx] = False
        context_X = self.X[context_mask]
        context_y = self.y[context_mask]
        target_X = self.X[~context_mask]
        target_y = self.y[~context_mask]
        return context_X, context_y, target_X, target_y

class IMContextTransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=64, n_layer=2, n_head=2):
        super().__init__()
        config = GPT2Config(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=n_positions,
            n_ctx=n_positions,
            vocab_size=1,
        )
        self.embed = nn.Linear(n_dims + 1, n_embd)
        self.transformer = GPT2Model(config)
        self.read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs, ys):
        # xs: [N, D], ys: [N]
        ys = ys[..., None]
        seq = torch.cat([xs, ys], dim=-1)  # [N, D+1]
        return seq

    def forward(self, context_x, context_y, target_x):
        # context_x: [B, N_ctx, D], context_y: [B, N_ctx], target_x: [B, 1, D]
        B, N_ctx, D = context_x.shape
        # Combine context
        seq = self._combine(context_x, context_y)  # [B, N_ctx, D+1]
        emb = self.embed(seq)  # [B, N_ctx, n_embd]
        out = self.transformer(inputs_embeds=emb).last_hidden_state  # [B, N_ctx, n_embd]
        # Use mean pooling over context
        pooled = out.mean(dim=1)  # [B, n_embd]
        # Predict target
        target_emb = self.embed(torch.cat([target_x, torch.zeros_like(target_x[..., :1])], dim=-1)).squeeze(1)  # [B, n_embd]
        combined = pooled + target_emb  # [B, n_embd]
        pred = self.read_out(combined).squeeze(-1)  # [B]
        return pred

def run_gpt2_imcontext_regression():
    print("Running GPT2 regression (im-context logic)...")
    train_path = "../data/regression/ALL_pca_train.csv"
    test_path = "../data/regression/ALL_pca_test.csv"

    # Load and scale data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.drop(columns=['age']).values
    y_train = df_train['age'].values
    X_test = df_test.drop(columns=['age']).values
    y_test = df_test['age'].values

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Dataset and loader
    train_ds = IMContextTabularDataset(X_train, y_train_scaled)
    test_ds = IMContextTabularDataset(X_test, y_test_scaled)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_dims = X_train.shape[1]
    n_positions = X_train.shape[0]  # or set to max context size you want
    model = IMContextTransformerModel(n_dims=n_dims, n_positions=n_positions, n_embd=64, n_layer=2, n_head=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training
    model.train()
    for epoch in range(10):
        total_loss = 0
        for context_x, context_y, target_x, target_y in train_loader:
            context_x, context_y = context_x.to(device), context_y.to(device)
            target_x, target_y = target_x.to(device), target_y.to(device)
            optimizer.zero_grad()
            pred = model(context_x, context_y, target_x)
            loss = loss_fn(pred, target_y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * context_x.size(0)
        print(f"Epoch {epoch+1}/10 - Train Loss: {total_loss/len(train_ds):.4f}")

    # Evaluation
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for context_x, context_y, target_x, target_y in test_loader:
            context_x, context_y = context_x.to(device), context_y.to(device)
            target_x, target_y = target_x.to(device), target_y.to(device)
            pred = model(context_x, context_y, target_x).cpu().numpy()
            preds.append(pred)
            targets.append(target_y.cpu().numpy())
    y_pred_scaled = np.concatenate(preds)
    y_true_scaled = np.concatenate(targets)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"IM-Context GPT2 Regression Results - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

from sklearn.model_selection import ParameterGrid

def run_gpt2_imcontext_regression_tuning():
    print("Running GPT2 (im-context) regression hyperparameter tuning...")
    train_path = "../data/regression/ALL_pca_train.csv"
    test_path = "../data/regression/ALL_pca_test.csv"

    # Load and scale data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.drop(columns=['age']).values
    y_train = df_train['age'].values
    X_test = df_test.drop(columns=['age']).values
    y_test = df_test['age'].values

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Hyperparameter grid
    param_grid = {
        "n_embd": [32, 64],
        "n_layer": [1, 2],
        "n_head": [2, 4],
        "lr": [1e-3, 5e-4],
        "batch_size": [8, 16]
    }
    best_r2 = -np.inf
    best_params = None
    best_metrics = None

    for params in ParameterGrid(param_grid):
        print(f"Testing params: {params}")
        # Dataset and loader
        train_ds = IMContextTabularDataset(X_train, y_train_scaled)
        test_ds = IMContextTabularDataset(X_test, y_test_scaled)
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=params["batch_size"])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_dims = X_train.shape[1]
        n_positions = X_train.shape[0]
        model = IMContextTransformerModel(
            n_dims=n_dims,
            n_positions=n_positions,
            n_embd=params["n_embd"],
            n_layer=params["n_layer"],
            n_head=params["n_head"]
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        loss_fn = nn.MSELoss()

        # Training
        model.train()
        for epoch in range(5):  # Fewer epochs for tuning speed
            total_loss = 0
            for context_x, context_y, target_x, target_y in train_loader:
                context_x, context_y = context_x.to(device), context_y.to(device)
                target_x, target_y = target_x.to(device), target_y.to(device)
                optimizer.zero_grad()
                pred = model(context_x, context_y, target_x)
                loss = loss_fn(pred, target_y.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * context_x.size(0)
            print(f"Epoch {epoch+1}/5 - Train Loss: {total_loss/len(train_ds):.4f}")

        # Evaluation
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for context_x, context_y, target_x, target_y in test_loader:
                context_x, context_y = context_x.to(device), context_y.to(device)
                target_x, target_y = target_x.to(device), target_y.to(device)
                pred = model(context_x, context_y, target_x).cpu().numpy()
                preds.append(pred)
                targets.append(target_y.cpu().numpy())
        y_pred_scaled = np.concatenate(preds)
        y_true_scaled = np.concatenate(targets)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"Params {params} -> MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

        if r2 > best_r2:
            best_r2 = r2
            best_params = params
            best_metrics = (mae, mse, r2)

    print(f"\nBest params: {best_params}")
    print(f"Best results - MAE: {best_metrics[0]:.2f}, MSE: {best_metrics[1]:.2f}, R2: {best_metrics[2]:.2f}")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import GPT2Model, GPT2Config
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class IMContextTabularClassificationDataset(Dataset):
    """
    Each item is a (context_X, context_y, target_X, target_y) tuple.
    For simplicity, use all but one row as context, one as target.
    """
    def __init__(self, X, y, context_size=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.N = X.shape[0]
        self.context_size = context_size or (self.N - 1)
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        context_mask = np.ones(self.N, dtype=bool)
        context_mask[idx] = False
        context_X = self.X[context_mask]
        context_y = self.y[context_mask]
        target_X = self.X[~context_mask]
        target_y = self.y[~context_mask]
        return context_X, context_y, target_X, target_y

class IMContextTransformerClassifier(nn.Module):
    def __init__(self, n_dims, n_positions, n_classes, n_embd=64, n_layer=2, n_head=2):
        super().__init__()
        config = GPT2Config(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=n_positions,
            n_ctx=n_positions,
            vocab_size=1,
        )
        self.embed = nn.Linear(n_dims + 1, n_embd)
        self.transformer = GPT2Model(config)
        self.read_out = nn.Linear(n_embd, n_classes)

    @staticmethod
    def _combine(xs, ys):
        # xs: [N, D], ys: [N]
        ys = ys[..., None].float()
        seq = torch.cat([xs, ys], dim=-1)  # [N, D+1]
        return seq

    def forward(self, context_x, context_y, target_x):
        # context_x: [B, N_ctx, D], context_y: [B, N_ctx], target_x: [B, 1, D]
        B, N_ctx, D = context_x.shape
        seqs = []
        for b in range(B):
            seqs.append(self._combine(context_x[b], context_y[b]))
        seq = torch.stack(seqs)  # [B, N_ctx, D+1]
        emb = self.embed(seq)  # [B, N_ctx, n_embd]
        out = self.transformer(inputs_embeds=emb).last_hidden_state  # [B, N_ctx, n_embd]
        pooled = out.mean(dim=1)  # [B, n_embd]
        # Target embedding (with dummy label 0)
        target_emb = self.embed(torch.cat([target_x, torch.zeros_like(target_x[..., :1])], dim=-1)).squeeze(1)  # [B, n_embd]
        combined = pooled + target_emb  # [B, n_embd]
        logits = self.read_out(combined)  # [B, n_classes]
        return logits

def run_gpt2_imcontext_classification():
    print("Running GPT2 classification (im-context logic)...")
    train_path = "../data/classification//cancer/ALL_pca_train.csv"
    test_path = "../data/classification//cancer/ALL_pca_test.csv"

    # Load and scale data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.drop(columns=['target']).values
    y_train = df_train['target'].values
    X_test = df_test.drop(columns=['target']).values
    y_test = df_test['target'].values

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)

    scaler_X = StandardScaler().fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Dataset and loader
    train_ds = IMContextTabularClassificationDataset(X_train, y_train_enc)
    test_ds = IMContextTabularClassificationDataset(X_test, y_test_enc)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_dims = X_train.shape[1]
    n_positions = X_train.shape[0]
    model = IMContextTransformerClassifier(n_dims=n_dims, n_positions=n_positions, n_classes=n_classes, n_embd=64, n_layer=2, n_head=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training
    model.train()
    for epoch in range(10):
        total_loss = 0
        for context_x, context_y, target_x, target_y in train_loader:
            context_x, context_y = context_x.to(device), context_y.to(device)
            target_x, target_y = target_x.to(device), target_y.to(device)
            optimizer.zero_grad()
            logits = model(context_x, context_y, target_x)
            loss = loss_fn(logits, target_y.squeeze().long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * context_x.size(0)
        print(f"Epoch {epoch+1}/10 - Train Loss: {total_loss/len(train_ds):.4f}")

    # Evaluation
    model.eval()
    preds = []
    targets = []
    probs = []
    with torch.no_grad():
        for context_x, context_y, target_x, target_y in test_loader:
            context_x, context_y = context_x.to(device), context_y.to(device)
            target_x, target_y = target_x.to(device), target_y.to(device)
            logits = model(context_x, context_y, target_x)
            prob = torch.softmax(logits, dim=-1)
            pred = torch.argmax(prob, dim=-1)
            preds.append(pred.cpu().numpy())
            targets.append(target_y.cpu().numpy())
            probs.append(prob.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets).flatten()
    y_prob = np.concatenate(probs)

    acc = accuracy_score(y_true, y_pred)
    print(f"IM-Context GPT2 Classification Results - Accuracy: {acc:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'IM-Context GPT2 Classification Confusion Matrix\nAccuracy: {acc:.2f}')
    plt.savefig('outputs/gpt2_imcontext_classification_cm.png')
    plt.show()

    # ROC curve (only for binary)
    if n_classes == 2:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for IM-Context GPT2')
        plt.legend(loc='lower right')
        plt.savefig('outputs/gpt2_imcontext_classification_roc.png')
        plt.show()
        print(f"ROC AUC: {roc_auc:.2f}")

from sklearn.model_selection import ParameterGrid

def run_gpt2_imcontext_classification_tuning():
    print("Running GPT2 (im-context) classification hyperparameter tuning...")
    train_path = "../data/classification/cancer/ALL_pca_train.csv"
    test_path = "../data/classification/cancer/ALL_pca_test.csv"

    # Load and scale data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.drop(columns=['target']).values
    y_train = df_train['target'].values
    X_test = df_test.drop(columns=['target']).values
    y_test = df_test['target'].values

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)

    scaler_X = StandardScaler().fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Hyperparameter grid
    param_grid = {
        "n_embd": [32, 64],
        "n_layer": [1, 2],
        "n_head": [2, 4],
        "lr": [1e-3, 5e-4],
        "batch_size": [8, 16]
    }
    best_acc = -np.inf
    best_params = None
    best_metrics = None

    for params in ParameterGrid(param_grid):
        print(f"Testing params: {params}")
        # Dataset and loader
        train_ds = IMContextTabularClassificationDataset(X_train, y_train_enc)
        test_ds = IMContextTabularClassificationDataset(X_test, y_test_enc)
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=params["batch_size"])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_dims = X_train.shape[1]
        n_positions = X_train.shape[0]
        model = IMContextTransformerClassifier(
            n_dims=n_dims,
            n_positions=n_positions,
            n_classes=n_classes,
            n_embd=params["n_embd"],
            n_layer=params["n_layer"],
            n_head=params["n_head"]
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        loss_fn = nn.CrossEntropyLoss()

        # Training
        model.train()
        for epoch in range(5):  # Fewer epochs for tuning speed
            total_loss = 0
            for context_x, context_y, target_x, target_y in train_loader:
                context_x, context_y = context_x.to(device), context_y.to(device)
                target_x, target_y = target_x.to(device), target_y.to(device)
                optimizer.zero_grad()
                logits = model(context_x, context_y, target_x)
                loss = loss_fn(logits, target_y.squeeze().long())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * context_x.size(0)
            print(f"Epoch {epoch+1}/5 - Train Loss: {total_loss/len(train_ds):.4f}")

        # Evaluation
        model.eval()
        preds = []
        targets = []
        probs = []
        with torch.no_grad():
            for context_x, context_y, target_x, target_y in test_loader:
                context_x, context_y = context_x.to(device), context_y.to(device)
                target_x, target_y = target_x.to(device), target_y.to(device)
                logits = model(context_x, context_y, target_x)
                prob = torch.softmax(logits, dim=-1)
                pred = torch.argmax(prob, dim=-1)
                preds.append(pred.cpu().numpy())
                targets.append(target_y.cpu().numpy())
                probs.append(prob.cpu().numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(targets).flatten()
        y_prob = np.concatenate(probs)

        acc = accuracy_score(y_true, y_pred)
        print(f"Params {params} -> Accuracy: {acc:.2f}")

        # ROC AUC for binary
        roc_auc = None
        if n_classes == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            print(f"ROC AUC: {roc_auc:.2f}")

        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_metrics = (acc, roc_auc)

    print(f"\nBest params: {best_params}")
    print(f"Best results - Accuracy: {best_metrics[0]:.2f}" + (f", ROC AUC: {best_metrics[1]:.2f}" if best_metrics[1] is not None else ""))


if __name__ == "__main__":
    run_gpt2_imcontext_classification_tuning()

