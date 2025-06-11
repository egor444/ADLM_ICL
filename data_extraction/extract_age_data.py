
import pandas as pd
import os
import time
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#### Radiomics extraction, required for other functions ###
def extract_radiomics_data():
    ''' Extract all radiomics data from the subfolders into a singler file for each type (wat and fat)'''
    print("Running radiomics data extraction")
    paths = pd.read_csv('../paths.csv', index_col=0)
    radiomics_path = paths.loc["radiomics"].iloc[0]

    # only use neessesary eids
    cancer_eids = pd.read_csv("/vol/miltank/projects/ukbb/projects/practical_ss25_icl/disease_filtered_data_5year/cancer_timerange_5year.csv", usecols=["eid"])["eid"].unique().tolist()
    healthy_train_eids = pd.read_csv("/vol/miltank/projects/ukbb/projects/practical_ss25_icl/whole_body_3d_healthy_noselfreported_noicd10_assessment2_train_df.csv", usecols=["eid"])["eid"].unique().tolist()
    healthy_val_eids = pd.read_csv("/vol/miltank/projects/ukbb/projects/practical_ss25_icl/whole_body_3d_healthy_noselfreported_noicd10_assessment2_val_df.csv", usecols=["eid"])["eid"].unique().tolist()
    eids = list(set(cancer_eids) | set(healthy_train_eids) | set(healthy_val_eids))
    eids = [str(eid) for eid in eids]  # ensure all EIDs are strings

    print(f"Radiomics: Extracting {len(eids)} EIDs. Reading first radiomics data")
    radiomics_wat = pd.read_csv(radiomics_path + eids[0] + "/radiomics_features_wat.csv")
    radiomics_wat["eid"] = eids[0]
    radiomics_fat = pd.read_csv(radiomics_path + eids[0] + "/radiomics_features_fat.csv")
    radiomics_fat["eid"] = eids[0]
    print("Radiomics: Initialized, starting loop.")
    time_start = time.time()
    save_iterations = 1 
    for i in range(1, len(eids)):
        if i == 11:
            time_end = time.time()
            print(f"##Radiomics: Time taken for first 10 iterations: {(time_end - time_start):.2f} seconds")
            time_approx = ((time_end - time_start) * len(eids) / 10) * 3  # estimate time for all iterations, assuming 3x the time of first 10 iterations
            # time_approx = len(eids) * 100 # estimate time for all iterations, assuming 100 seconds per iteration
            time_name = "seconds"
            if time_approx > 60:
                time_approx = time_approx / 60
                time_name = "minutes"
            elif time_approx > 3600:
                time_approx = time_approx / 3600
                time_name = "hours"
            print(f"##Radiomics: Time approximation for all iterations: {time_approx:.2f} {time_name}")
        if (time.time() - time_start) > (60 * 5 * save_iterations):  # save every 5 minutes
            print(f"Radiomics: Extracted {i} / {len(eids)}, time taken: {(time.time() - time_start):.2f} seconds.\n\tSaving progress...")
            radiomics_wat.to_csv("../data/radiomics_wat.csv", index=False)
            radiomics_fat.to_csv("../data/radiomics_fat.csv", index=False)
            print(f"\tSave {save_iterations}. Dataframe size: {radiomics_wat.memory_usage(deep=True).sum() / 1e9:.2f} GB")
            save_iterations += 1
        radiomics_wat_temp = pd.read_csv(radiomics_path + eids[i] + "/radiomics_features_wat.csv")
        radiomics_fat_temp = pd.read_csv(radiomics_path + eids[i] + "/radiomics_features_fat.csv")

        radiomics_wat_temp["eid"] = eids[i]
        radiomics_fat_temp["eid"] = eids[i]

        radiomics_wat = pd.concat([radiomics_wat, radiomics_wat_temp], axis=0)
        radiomics_fat = pd.concat([radiomics_fat, radiomics_fat_temp], axis=0)
    

    radiomics_wat = radiomics_wat.reset_index(drop=True)
    radiomics_fat = radiomics_fat.reset_index(drop=True)

    radiomics_wat.to_csv("../data/radiomics_wat.csv", index=False)
    radiomics_fat.to_csv("../data/radiomics_fat.csv", index=False)
    print("Radiomics: DONE Radiomics")

########## FUNCTIONS FOR REGRESSION DATA EXTRACTION ##########

# 1
def merge_embeddings_and_reg_data():
    """Extracts age data from the healthy train and val datasets and merges them with the embeddings dataset."""

    print("REG 1: Running age data extraction")
    paths = pd.read_csv('../paths.csv', index_col=0)
    healthy_t = pd.read_csv(paths.loc["healthy_train"].iloc[0])
    healthy_v = pd.read_csv(paths.loc["healthy_val"].iloc[0])
    embeddings = pd.read_csv("../data/other/embeddings_cls.csv")

    column_names_final = [f"feature_{i}" for i in range(1025)] + ["eid","age"]

    print("\tMerging data")
    # merge by eid, 
    healthy_t_merged = healthy_t.merge(embeddings, on="eid", how="inner", suffixes=("", "_y"))
    healthy_v_merged = healthy_v.merge(embeddings, on="eid", how="inner", suffixes=("", "_y"))

    print("\textracting columns")
    healthy_t_merged = healthy_t_merged[column_names_final]
    healthy_v_merged = healthy_v_merged[column_names_final]

    print("\tsaving data...")
    # save the dataframes
    healthy_t_merged.to_csv("../data/other/emb_age_healthy_train.csv", index=False)
    healthy_v_merged.to_csv("../data/other/emb_age_healthy_test.csv", index=False)

    print("Age Data: DONE Age data extraction")

# 2
def merge_radiomics_and_embeddings_reg(separate_types=False):
    """Merges the radiomics data with the age+embeddings data."""
    print("REG 2: Starting combining radiomics and age data of healthy patients")
    rad_types = ["wat", "fat"]
    set_types = ["train", "test"]
    if separate_types:
        for rad_type in rad_types:
            print(f"\tLoading radiomics {rad_type} data")
            radiomics = pd.read_csv(f"../data/other/radiomics_{rad_type}.csv")
            for set_type in set_types:
                print(f"\tLoading age data for {set_type}")
                mae_age_data = pd.read_csv(f"../data/other/emb_age_healthy_{set_type}.csv", usecols=["eid", "age"])
                print(f"\tMerging radiomics {rad_type} data with age {set_type} data")
                mae_age_data = radiomics.merge(mae_age_data, on="eid", how="inner", suffixes=("", "_y"))
                mae_age_data = mae_age_data.loc[:, ~mae_age_data.columns.str.endswith("_y")]
                mae_age_data.to_csv(f"../data/other/rad_emb_healthy_{rad_type}_{set_type}.csv", index=False)
                print(f"\tSaved radiomics {rad_type} data with age {set_type} data; size: {mae_age_data.shape}")   
    else:
        print(f"\tLoading radiomics data")
        radiomics_fat = pd.read_csv("../data/other/radiomics_fat.csv")
        radiomics_wat = pd.read_csv("../data/other/radiomics_wat.csv")
        # add type to columns
        radiomics_fat.rename(columns=lambda x: f"{x}_fat" if x not in ["eid", "age"] else x, inplace=True)
        radiomics_wat.rename(columns=lambda x: f"{x}_wat" if x not in ["eid", "age"] else x, inplace=True)

        for set_type in set_types:
            print(f"\tLoading age data for {set_type}")
            mae_age_data = pd.read_csv(f"../data/other/emb_age_healthy_{set_type}.csv", usecols=["eid", "age"])
            print(f"\tMerging radiomics data with age {set_type} data")
            mae_age_data = radiomics_fat.merge(mae_age_data, on="eid", how="inner", suffixes=("", "_y"))
            mae_age_data = mae_age_data.merge(radiomics_wat, on="eid", how="inner", suffixes=("", "_y"))
            mae_age_data = mae_age_data.loc[:, ~mae_age_data.columns.str.endswith("_y")]
            mae_age_data.to_csv(f"../data/other/rad_emb_healthy_ALL_{set_type}.csv", index=False)
            print(f"\tSaved radiomics data with age {set_type} data; size: {mae_age_data.shape}")
    
    print("Rad&Age: DONE Merging radiomics and age data")

# 3
def create_regression_data(separate_types=False):
    print("REG 3: Creating regression data files")
    rad_types = ["wat", "fat"] if separate_types else ["ALL"]
    set_types = ["train", "test"]
    na_cols = [] # keep the same na columns for all datasets for consistency

    for rad_type in rad_types:
        for set_type in set_types:
            print(f"\tLoading {set_type} data for {rad_type}")
            data = pd.read_csv(f"../data/other/rad_emb_healthy_{rad_type}_{set_type}.csv")
            print(f"\tData loaded, cleaning data")
            if not na_cols:
                # drop na columns with more than 30% missing values, then rows with any na values
                n = 0.3
                na_cols = data.columns[data.isna().mean() >  n].values.tolist()
            data = data.drop(columns=na_cols)
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.dropna()
            # drop non number columns 
            non_float_cols = data.select_dtypes(exclude=[np.float64, np.int64]).columns
            data = data.drop(columns=non_float_cols)
            # normalize features
            cols = data.columns.tolist()
            cols.remove('eid')
            cols.remove('age')
            data[cols] = (data[cols] - data[cols].mean()) / data[cols].std()  # normalize features
            print(f"\tSaving {set_type} data for {rad_type}")
            data.to_csv(f"../data/regression/{set_type}_{rad_type}.csv", index=False)
    print("Reg: DONE Creating regression data files")

########## FUNCTIONS FOR CLASSIFICATION DATA EXTRACTION ##########

# 1
def merge_embeddings_and_class_data():
    """Merges the embeddings data with the age data."""
    print("CLASS 1: Starting combining embeddings and time to event data of cancer patients")
    path = "../data/other/cancer_timerange_5year.csv"
    cancer_data = pd.read_csv(path, usecols=["eid"])
    embeddings = pd.read_csv("../data/other/embeddings_cls.csv")
    print("\tMerging embeddings and cancer data")
    cancer_data = cancer_data.merge(embeddings, on="eid", how="inner", suffixes=("", "_y"))
    cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.endswith("_y")]
    print("\tSaving merged data")
    cancer_data.to_csv("../data/other/emb_cancer_5year.csv", index=False)

# 2
def merge_radiomics_and_embeddings_class(separate_types=False):
    """Merges the radiomics data with the embeddings and time to event data."""
    print("CLASS 2: Starting combining radiomics, embeddings and time to event data of cancer patients")
    if separate_types:
        rad_type = "fat"
        print(f"\tLoading radiomics {rad_type} data")
        radiomics = pd.read_csv(f"../data/other/radiomics_{rad_type}.csv")
        print("\tLoading embeddings and time to event data")
        cancer_emb_data = pd.read_csv("../data/other/emb_cancer_5year.csv")
        print("\tMerging radiomics and cancer data")
        cancer_data = radiomics.merge(cancer_emb_data, on="eid", how="inner", suffixes=("", "_y"))
        cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.endswith("_y")]
        print("\tSaving merged data")
        cancer_data.to_csv("../data/other/rad_emb_cancer_5year.csv", index=False)
    else:
        print(f"\tLoading radiomics data")
        radiomics_fat = pd.read_csv("../data/other/radiomics_fat.csv")
        radiomics_wat = pd.read_csv("../data/other/radiomics_wat.csv")
        # add type to columns
        radiomics_fat.rename(columns=lambda x: f"{x}_fat" if x not in ["eid"] else x, inplace=True)
        radiomics_wat.rename(columns=lambda x: f"{x}_wat" if x not in ["eid"] else x, inplace=True)

        print("\tLoading embeddings and time to event data")
        cancer_emb_data = pd.read_csv("../data/other/emb_cancer_5year.csv")
        print("\tMerging radiomics and cancer data")
        cancer_data = radiomics_fat.merge(cancer_emb_data, on="eid", how="inner", suffixes=("", "_y"))
        cancer_data = cancer_data.merge(radiomics_wat, on="eid", how="inner", suffixes=("", "_y"))
        cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.endswith("_y")]
        print("\tSaving merged data")
        cancer_data.to_csv("../data/other/rad_emb_cancer_5year.csv", index=False)

# 3 WARNING: This requires merge_radiomics_and_embeddings_reg to be run 
def combine_rad_emb_healthy_and_cancer(separate_types=False):
    """Combines the healthy and cancer data for classification."""
    print("CLASS 3: Combining healthy and cancer data for classification")
    print(f"\tLoading healthy data")
    if separate_types:
        healthy_train = pd.read_csv("../data/other/rad_emb_healthy_fat_train.csv")
        healthy_test = pd.read_csv("../data/other/rad_emb_healthy_fat_test.csv")
    else:
        healthy_train = pd.read_csv("../data/other/rad_emb_healthy_ALL_train.csv")
        healthy_test = pd.read_csv("../data/other/rad_emb_healthy_ALL_test.csv")
    healthy_combined = pd.concat([healthy_train, healthy_test], axis=0)
    healthy_combined["target"] = 0  # healthy patients are labeled as 0
    del healthy_train, healthy_test # free memory
    print(f"\tLoading cancer data")
    cancer_data = pd.read_csv("../data/other/rad_emb_cancer_5year.csv")
    cancer_data["target"] = 1  # cancer patients are labeled as 1
    print("\tCombining healthy and cancer data")
    combined_data = pd.concat([healthy_combined, cancer_data], axis=0)
    print("\tSaving combined data")
    if separate_types:
        combined_data.to_csv("../data/other/rad_emb_combined_fat.csv", index=False)
    else:
        combined_data.to_csv("../data/other/rad_emb_combined_ALL.csv", index=False)

# 4
def create_classification_data(separate_types=False):
    """Creates the classification data files."""
    print("CLASS 4: Creating classification data files")
    print("\tLoading combined data for classification")
    data_path_type = "fat" if separate_types else "ALL"
    combined_data = pd.read_csv(f"../data/other/rad_emb_combined_{data_path_type}.csv")
    print("\tCleaning data")
    n = 0.3
    na_cols = combined_data.columns[combined_data.isna().mean() > n]
    combined_data = combined_data.drop(columns=na_cols)
    combined_data = combined_data.replace([np.inf, -np.inf], np.nan)
    combined_data = combined_data.dropna()
    # drop non number columns
    non_float_cols = combined_data.select_dtypes(exclude=[np.float64, np.int64]).columns
    combined_data = combined_data.drop(columns=non_float_cols)
    # normalize features
    cols = combined_data.columns.tolist()
    cols.remove('eid')
    cols.remove('target')
    combined_data[cols] = (combined_data[cols] - combined_data[cols].mean()) / combined_data[cols].std() 
    print("\tSplitting data into train and test sets")
    train_data, test_data = train_test_split(combined_data, test_size=0.2, stratify=combined_data['target'], random_state=42, shuffle=True)
    print("\tSaving train and test data")
    train_data.to_csv(f"../data/classification/train_{data_path_type}.csv", index=False)
    test_data.to_csv(f"../data/classification/test_{data_path_type}.csv", index=False)

### Function for pca

def perform_pca(data_path_train, data_path_val, output_path, ignore_cols=None, n_components=0.95):
    """Performs PCA on the training and validation data and saves the transformed data."""
    print("PCA: Starting PCA on data")
    print(f"\tLoading training data")
    data = pd.read_csv(data_path_train)
    if ignore_cols is not None:
        target_data = data[ignore_cols]
        data = data.drop(columns=ignore_cols)
    pca = PCA(n_components=n_components)
    print(f"\tFitting PCA on training data")
    pca.fit(data)
    pca_data_train = pca.transform(data)
    pca_data_train = pd.DataFrame(pca_data_train, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    if ignore_cols is not None:
        pca_data_train[ignore_cols] = target_data
    pca_data_train.to_csv(output_path + "pca_train.csv", index=False)
    print(f"\tPCA applied, total features: {pca_data_train.shape[1]-len(ignore_cols)}")
    data = pd.read_csv(data_path_val)
    if ignore_cols is not None:
        target_data = data[ignore_cols]
        data = data.drop(columns=ignore_cols)
    print(f"\tTransforming validation data with PCA")
    pca_data_val = pca.transform(data)
    pca_data_val = pd.DataFrame(pca_data_val, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    if ignore_cols is not None:
        pca_data_val[ignore_cols] = target_data
    pca_data_val.to_csv(output_path + "pca_val.csv", index=False)
    print(f"\tPCA applied, total features: {pca_data_val.shape[1]-len(ignore_cols)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract or change data")
    parser.add_argument("--radiomics", action="store_true", help="Extract radiomics data")

    # arguments for regression data extraction
    parser.add_argument("--run_reg", action="store_true", help="Run regression data extraction")
    parser.add_argument("--emb_age_reg", action="store_true", help="Extract embeddings and age data")
    parser.add_argument("--rad_emb_reg", action="store_true", help="Merge radiomics and age data")
    parser.add_argument("--create_reg", action="store_true", help="Create regression data files from radiomics and embeddings data")
    parser.add_argument("--pca_reg", action="store_true", help="Perform PCA on regression data")
    
    # arguments for classification data extraction
    parser.add_argument("--run_class", action="store_true", help="Run classification data extraction")
    parser.add_argument("--emb_tte_class", action="store_true", help="Extract embeddings and time to event data for classification")
    parser.add_argument("--rad_emb_class", action="store_true", help="Merge radiomics and embeddings data for classification")
    parser.add_argument("--combine_rad_emb_healthy_and_cancer", action="store_true", help="Combine healthy and cancer data for classification")
    parser.add_argument("--create_class", action="store_true", help="Create classification data files from combined data")
    parser.add_argument("--pca_class", action="store_true", help="Perform PCA on classification data")
    
    args = parser.parse_args()

    print("###### STARTING #######")
    total_operations = sum([arg for arg in vars(args).values() if isinstance(arg, bool) and arg])
    if args.run_reg:
        total_operations += 3
    if args.run_class:
        total_operations += 4
    print(f"#### Total operations to run: {total_operations}")

    with open('extra_cols.csv', 'r') as f:
        extra_cols = f.read().splitlines()
    extra_cols = [col.strip() for col in extra_cols if col.strip()]  
    extra_cols = [col.split(",") for col in extra_cols]  
    field_names = [col[0] for col in extra_cols]
    field_replacement_names = [col[1] for col in extra_cols]

    if args.radiomics: # if this is run, the old extracted data would be lost (and it takes some time)
        pass
        extract_radiomics_data()

    if args.run_reg:
        merge_embeddings_and_reg_data()
        merge_radiomics_and_embeddings_reg()
        create_regression_data()
        perform_pca("../data/regression/train_ALL.csv", "../data/regression/test_ALL.csv", "../data/regression/ALL_", ignore_cols=["eid", "age"], n_components=0.95)
    else:
        if args.emb_age_reg:
            merge_embeddings_and_reg_data()
        if args.rad_emb_reg:
            merge_radiomics_and_embeddings_reg()
        if args.create_reg:
            create_regression_data()
        if args.pca_reg:
            perform_pca("../data/regression/train_ALL.csv", "../data/regression/test_ALL.csv", "../data/regression/ALL_", ignore_cols=["eid", "age"], n_components=0.95)

    if args.run_class:
        merge_embeddings_and_class_data()
        merge_radiomics_and_embeddings_class()
        combine_rad_emb_healthy_and_cancer()
        create_classification_data()
        perform_pca("../data/classification/train_ALL.csv", "../data/classification/test_ALL.csv", "../data/classification/ALL_", ignore_cols=["eid", "target"], n_components=0.95)
    else:
        if args.emb_tte_class:
            merge_embeddings_and_class_data()
        if args.rad_emb_class:
            merge_radiomics_and_embeddings_class()
        if args.combine_rad_emb_healthy_and_cancer:
            combine_rad_emb_healthy_and_cancer()
        if args.create_class:
            create_classification_data()
        if args.pca_class:
            perform_pca("../data/classification/train_ALL.csv", "../data/classification/test_ALL.csv", "../data/classification/ALL_", ignore_cols=["eid", "target"], n_components=0.95)

    print("DONE ALL")

