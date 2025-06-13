import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import mutual_info_regression

import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.decomposition import PCA


def run_data_analysis():
    """Run the data analysis on the radiomics data by fitting regression models."""
    print("### Starting data analysis on radiomics/embeddings data ###")

    data = pd.read_csv("../data/radiomics_embeddings_wat.csv")

    # Remove 'eid' column
    
    data = data.drop(columns=['eid', 'unused_exception']) # last column is non float
    
    # drop na columns with more than x% missing values
    x = 0.3
    na_cols = data.columns[data.isna().mean() > x]
    data_nona = data.drop(columns=na_cols)
    data = data_nona.dropna()
    
    print(f"Total data samples: {data.shape}")
    X = data.drop(columns=["age"])
    y = data["age"]

    print("Data shape:", X.shape, y.shape)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data standardized")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # Fit a linear regression model
        model = LinearRegression(device='cuda')
        model.fit(X_train_scaled, y_train)
        print("Model fitted")
        # Make predictions
        y_pred_lr = model.predict(X_test_scaled)
        
        print("Predictions made")
        # Evaluate the model
        lr_mae = mean_absolute_error(y_test, y_pred_lr)
        lr_r2 = r2_score(y_test, y_pred_lr)
        print("LR: Mean Absolute Error:", lr_mae)
        print("LR: R^2 Score:", lr_r2)
        ## Fit a Lasso regression model
        lasso = LassoCV(cv=5, random_state=42, device='cuda')
        lasso.fit(X_train_scaled, y_train)
        print("Lasso model fitted")
        # Make predictions
        y_pred_lasso = lasso.predict(X_test_scaled)
        print("Lasso predictions made")
        # Evaluate the model
        lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
        lasso_r2 = r2_score(y_test, y_pred_lasso)
        print("Lasso: Mean Absolute Error:", lasso_mae)
        print("Lasso: R^2 Score:", lasso_r2)

        # Fit an ElasticNet regression model
        elastic_net = ElasticNetCV(cv=5, random_state=42, device='cuda')
        elastic_net.fit(X_train_scaled, y_train)
        print("ElasticNet model fitted")
        # Make predictions
        y_pred_en = elastic_net.predict(X_test_scaled)
        print("ElasticNet predictions made")
        # Evaluate the model
        en_mae = mean_absolute_error(y_test, y_pred_en)
        en_r2 = r2_score(y_test, y_pred_en)
        print("ElasticNet: Mean Absolute Error:", en_mae)
        print("ElasticNet: R^2 Score:", en_r2)

    print("PLotting results...")
    alphas = np.logspace(-0.5, 1, 100)
    plt.plot(lasso.alphas_, np.count_nonzero(lasso.path(X, y, alphas=alphas, max_iter=100000)[1], axis=0))
    plt.xscale('log')
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Number of Non-zero Coefficients")
    plt.title("Lasso Path: Features vs Regularization")
    plt.grid(True)
    plt.savefig("../outfiles/lasso_path.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred_lasso, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Lasso Regression')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred_en, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('ElasticNet Regression')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_pred_lr, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Linear Regression')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("../outfiles/regression_results.png")
    plt.close()

    # top features by mutual information
    mi_scores = mutual_info_regression(X_train, y_train)
    mi_scores = pd.Series(mi_scores, index=data.drop(columns=["age"]).columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    mi_scores[:30].plot(kind='bar')
    plt.title('Mutual Information Scores for Features')
    plt.xlabel('Features')
    plt.ylabel('Mutual Information Score')
    plt.xticks(rotation=50)
    plt.tight_layout()
    plt.savefig("../outfiles/mi_scores.png")
    plt.close()
    print("Plots saved, saving lasso feature inforamtion")

    lasso_coef = pd.Series(lasso.coef_, index=data.drop(columns=["age"]).columns)
    print(f"Total Lasso features: {len(lasso_coef)}")
    lasso_coef = lasso_coef[lasso_coef != 0].sort_values(ascending=False)
    selected_lasso = lasso_coef[lasso_coef != 0].index.tolist()
    print(f"Selected {len(selected_lasso)} features by Lasso Regression")
    lasso_coef.to_csv("../outfiles/lasso_coefficients.csv", header=True)
    print("Lasso coefficients saved to ../outfiles/lasso_coefficients.csv")

    print("### Analysis complete ###")

def run_data_pca():
    """Run PCA on the radiomics data."""
    print("### Starting PCA on radiomics/embeddings data ###")
    data = pd.read_csv("../data/radiomics_wat.csv")
    
    age_data = pd.read_csv("/vol/miltank/projects/ukbb/data/whole_body/mae_embeddings/embeddings_cls.csv", usecols=['eid', 'age'])
    age_data = age_data.rename(columns={'21022': 'age'})

    data = data.merge(age_data, on='eid', how='left')
    print("Data loaded, shape:", data.shape)
    data = data.drop(columns=['unused_exception'])  
    x = 0.3
    na_cols = data.columns[data.isna().mean() > x]
    data_nona = data.drop(columns=na_cols)
    data = data_nona.dropna()
    eids = data['eid']
    age = data['age']
    data = data.drop(columns=['eid']) 

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print("Data standardized")
    # fit PCA
    pca = PCA(n_components=0.95)  
    pca.fit(data_scaled)
    print("PCA fitted")
    # Transform
    data_pca = pca.transform(data_scaled)
    # keep eids and age for later use
    data_pca = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])])
    data_pca['age'] = age.values
    data_pca['eid'] = eids.values
    data_pca = data_pca.set_index('eid')
    # Save PCA components
    data_pca.to_csv("./outfiles/pca_components.csv")
    print("Data transformed using PCA")
    print(f"Total PCA features: {data_pca.shape[1]}")

if __name__ == "__main__":
    #run_data_analysis()
    run_data_pca()
    

   

    