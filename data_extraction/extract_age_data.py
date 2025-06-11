
import pandas as pd
import os
import time
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

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

    print("Running age data extraction")
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
def merge_radiomics_and_embeddings_reg():
    """Merges the radiomics data with the age+embeddings data."""
    print("Starting combining radiomics and age data of healthy patients")
    rad_types = ["wat", "fat"]
    set_types = ["train", "test"]

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
    
    print("Rad&Age: DONE Merging radiomics and age data")

# 3
def create_regression_data():
    print("Creating regression data files")
    rad_types = ["wat", "fat"]
    set_types = ["train", "test"]
    na_cols = [] # keep the same na columns for all datasets for consistency
    for rad_type in rad_types:
        for set_type in set_types:
            print(f"\tLoading {set_type} data for {rad_type}")
            data = pd.read_csv(f"../data/other/rad_emb_healthy_{rad_type}_{set_type}.csv")
            print(f"\tData loaded, cleaning data")
            if na_cols == []:
                # drop na columns with more than 30% missing values, then rows with any na values
                n = 0.3
                na_cols = data.columns[data.isna().mean() >  n]
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
    print("Starting combining embeddings and time to event data of cancer patients")
    path = "../data/other/cancer_timerange_5year.csv"
    cancer_data = pd.read_csv(path, usecols=["eid"])
    embeddings = pd.read_csv("../data/other/embeddings_cls.csv")
    print("\tMerging embeddings and cancer data")
    cancer_data = cancer_data.merge(embeddings, on="eid", how="inner", suffixes=("", "_y"))
    cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.endswith("_y")]
    print("\tSaving merged data")
    cancer_data.to_csv("../data/other/emb_cancer_5year.csv", index=False)

# 2
def merge_radiomics_and_embeddings_class():
    """Merges the radiomics data with the embeddings and time to event data."""
    print("Starting combining radiomics, embeddings and time to event data of cancer patients")
    rad_type = "fat"
    print(f"\tLoading radiomics {rad_type} data")
    radiomics = pd.read_csv("../data/other/radiomics_fat.csv")
    print("\tLoading embeddings and time to event data")
    cancer_emb_data = pd.read_csv("../data/other/emb_cancer_5year.csv")
    print("\tMerging radiomics and cancer data")
    cancer_data = radiomics.merge(cancer_emb_data, on="eid", how="inner", suffixes=("", "_y"))
    cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.endswith("_y")]
    print("\tSaving merged data")
    cancer_data.to_csv("../data/other/rad_emb_cancer_5year.csv", index=False)

# 3 WARNING: This requires merge_radiomics_and_embeddings_reg to be run 
def combine_rad_emb_healthy_and_cancer():
    """Combines the healthy and cancer data for classification."""
    print("Combining healthy and cancer data for classification")
    print(f"\tLoading healthy data for fat")
    healthy_train = pd.read_csv("../data/other/rad_emb_healthy_fat_train.csv")
    healthy_test = pd.read_csv("../data/other/rad_emb_healthy_fat_test.csv")
    healthy_combined = pd.concat([healthy_train, healthy_test], axis=0)
    healthy_combined["target"] = 0  # healthy patients are labeled as 0
    del healthy_train, healthy_test # free memory
    print(f"\tLoading cancer data for fat")
    cancer_data = pd.read_csv("../data/other/rad_emb_cancer_5year.csv")
    cancer_data["target"] = 1  # cancer patients are labeled as 1
    print("\tCombining healthy and cancer data")
    combined_data = pd.concat([healthy_combined, cancer_data], axis=0)
    print("\tSaving combined data")
    combined_data.to_csv("../data/other/rad_emb_combined_fat.csv", index=False)

# 4
def create_classification_data():
    """Creates the classification data files."""
    print("Creating classification data files")
    print("\tLoading combined data for classification")
    combined_data = pd.read_csv("../data/other/rad_emb_combined_fat.csv")
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
    combined_data[cols] = (combined_data[cols] - combined_data[cols].mean()) / combined_data[cols].std()  # normalize features
    print("\tSplitting data into train and test sets")
    train_data, test_data = train_test_split(combined_data, test_size=0.2, stratify=combined_data['target'], random_state=42, shuffle=True)
    print("\tSaving train and test data")
    train_data.to_csv("../data/classification/train_fat.csv", index=False)
    test_data.to_csv("../data/classification/test_fat.csv", index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract or change data")
    parser.add_argument("--radiomics", action="store_true", help="Extract radiomics data")

    # arguments for regression data extraction
    parser.add_argument("--run_reg", action="store_true", help="Run regression data extraction")
    parser.add_argument("--emb_age_reg", action="store_true", help="Extract embeddings and age data")
    parser.add_argument("--rad_emb_reg", action="store_true", help="Merge radiomics and age data")
    parser.add_argument("--create_reg", action="store_true", help="Create regression data files from radiomics and embeddings data")
    
    # arguments for classification data extraction
    parser.add_argument("--run_class", action="store_true", help="Run classification data extraction")
    parser.add_argument("--emb_tte_class", action="store_true", help="Extract embeddings and time to event data for classification")
    parser.add_argument("--rad_emb_class", action="store_true", help="Merge radiomics and embeddings data for classification")
    parser.add_argument("--combine_rad_emb_healthy_and_cancer", action="store_true", help="Combine healthy and cancer data for classification")
    parser.add_argument("--create_class", action="store_true", help="Create classification data files from combined data")
    
    args = parser.parse_args()

    print("STARTING")

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
    else:
        if args.emb_age_reg:
            merge_embeddings_and_reg_data()
        if args.rad_emb_reg:
            merge_radiomics_and_embeddings_reg()
        if args.create_reg:
            create_regression_data()

    if args.run_class:
        merge_embeddings_and_class_data()
        merge_radiomics_and_embeddings_class()
        combine_rad_emb_healthy_and_cancer()
        create_classification_data()
    else:
        if args.emb_tte_class:
            merge_embeddings_and_class_data()
        if args.rad_emb_class:
            merge_radiomics_and_embeddings_class()
        if args.combine_rad_emb_healthy_and_cancer:
            combine_rad_emb_healthy_and_cancer()
        if args.create_class:
            create_classification_data()

    print("DONE ALL")

