
import pandas as pd
import os
import time
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import logging
import multiprocessing as mp
import sys
from data_manager import DataManager

def convert_time_to_string(seconds):
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(sec)}s"

#### Radiomics extraction, required for other functions ###
def extract_radiomics_data(eid_type="healthy", eid_paths=None, logger=None, data_path='/vol/miltank/projects/practical_sose25/in_context_learning/data/'):
    ''' Extract all radiomics data from the subfolders into a singler file for each type (wat and fat)'''
    if logger is None:
        logger = logging.getLogger("RADIOMICS")
    logger.info("Running radiomics data extraction")
    radiomics_path = '/vol/miltank/projects/ukbb/data/whole_body/radiomics/'

    if eid_paths is None or len(eid_paths) == 0:
        logger.error("Radiomics data paths not provided, cancelling extraction.")
        return

    output_paths = [f"{data_path}/raw/radiomics/radiomics_{eid_type}_fat.csv", f"{data_path}/raw/radiomics/radiomics_{eid_type}_wat.csv"]
    for output_path in output_paths:
        if os.path.exists(output_path):
            logger.info(f"Radiomics for {eid_type} data already extracted. Skipping extraction.")
            return
    
    if type(eid_paths) is str:
        eid_paths = [eid_paths]
    eids_lists = [pd.read_csv(eid_path, usecols=["eid"])["eid"].tolist() for eid_path in eid_paths]
    eids = list(set().union(*eids_lists))
    eids = [str(eid) for eid in eids]  # ensure all EIDs are strings

    logger.info(f"Extracting {len(eids)} EIDs. Reading first radiomics data")
    radiomics_wat = pd.read_csv(radiomics_path + eids[0] + "/radiomics_features_wat.csv")
    radiomics_wat["eid"] = eids[0]
    radiomics_fat = pd.read_csv(radiomics_path + eids[0] + "/radiomics_features_fat.csv")
    radiomics_fat["eid"] = eids[0]
    logger.info("Initialized, starting loop.")
    time_start = time.time()
    save_iterations = 1 
    for i in range(1, len(eids)):
        if i == 11:
            time_end = time.time()
            logger.info(f"\tTime taken for first 10 iterations: {(time_end - time_start):.2f} seconds")
            time_approx = ((time_end - time_start) * len(eids) / 10) * 3  # estimate time for all iterations, assuming 3x the time of first 10 iterations
            # time_approx = len(eids) * 100 # estimate time for all iterations, assuming 100 seconds per iteration
            time_name = convert_time_to_string(time_approx)
            logger.info(f"\tTime approximation for all iterations: {time_name}")
        if (time.time() - time_start) > (60 * 5 * save_iterations):  # save every 5 minutes
            logger.info(f"\tExtracted {i} / {len(eids)}, time taken: {convert_time_to_string(time.time() - time_start)}. Saving progress...")
            radiomics_wat.to_csv(output_paths[1], index=False)
            radiomics_fat.to_csv(output_paths[0], index=False)
            logger.info(f"\t\tSave {save_iterations}. Dataframe size: {radiomics_wat.memory_usage(deep=True).sum() / 1e9:.2f} GB")
            save_iterations += 1
        radiomics_wat_temp = pd.read_csv(radiomics_path + eids[i] + "/radiomics_features_wat.csv")
        radiomics_fat_temp = pd.read_csv(radiomics_path + eids[i] + "/radiomics_features_fat.csv")

        radiomics_wat_temp["eid"] = eids[i]
        radiomics_fat_temp["eid"] = eids[i]

        radiomics_wat = pd.concat([radiomics_wat, radiomics_wat_temp], axis=0)
        radiomics_fat = pd.concat([radiomics_fat, radiomics_fat_temp], axis=0)
    

    radiomics_wat = radiomics_wat.reset_index(drop=True)
    radiomics_fat = radiomics_fat.reset_index(drop=True)

    radiomics_wat.to_csv(output_paths[1], index=False)
    radiomics_fat.to_csv(output_paths[0], index=False)
    logger.info("Radiomics: DONE Radiomics")

########## OVERALL MERGING FUNCTIONS ##########

def merge_data_by_eid( data_paths=[], loaded_data=[], keepfile_indices=[], output_path=None, dropcols=[], only_cols=[]):
    """
    Merges the data by eid.
    args:
        data_paths (list): List of paths to the data files to be merged.
        output_path (str): Path where the merged data will be saved.
        keepfile_indices (list): List of indices of dataframes to keep in memory for further processing.
        loaded_data (list): List of dataframes that have been kept in memory.
        dropcols (list): List of columns to drop from the merged data.
    returns:
        keep_files (list): List of dataframes that have been kept in memory
    """
    logger = logging.getLogger("MERGE EIDS")
    logger.info(f"Starting merging {len(data_paths) + len(loaded_data)} files by eid")
    keep_files = []
    if len(data_paths) + len(loaded_data) <2:
        logger.error("Data less than 2 files to merge.")
        return
    if len(data_paths) > 0:
        data = pd.read_csv(data_paths[0])
        if len(keepfile_indices) > 0 and 0 in keepfile_indices:
            keep_files.append(data)
    else:
        data = loaded_data[0]
    if dropcols:
        data = data.drop(columns=dropcols)
    logger.info(f"Data 1 loaded, size: {data.shape}")
    for i, data_path in enumerate(data_paths[1:]):
        data_temp = pd.read_csv(data_path)
        if i+1 in keepfile_indices:
            keep_files.append(data_temp)
        if dropcols:
            data_temp = data_temp.drop(columns=dropcols)
        logger.info(f"Data {i + 2} loaded, size: {data_temp.shape}. Merging data")
        data = data.merge(data_temp, on="eid", how="inner", suffixes=("", "_y"))
        data = data.loc[:, ~data.columns.str.endswith("_y")]
    for i, keptfile in enumerate(loaded_data):
        logger.info(f"Kept file {i + 1} loaded, size: {keptfile.shape}. Merging data")
        if dropcols:
            keptfile = keptfile.drop(columns=dropcols)
        data = data.merge(keptfile, on="eid", how="inner", suffixes=("", "_y"))
        data = data.loc[:, ~data.columns.str.endswith("_y")]
    logger.info(f"Data merged, size: {data.shape}.")
    if only_cols:
        logger.info(f"Filtering data to only keep {len(only_cols)} columns")
        data = data[only_cols]
    if output_path:
        data.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    logger.info(f"Data saved, DONE merging")
    return data, keep_files


########## FUNCTIONS FOR REGRESSION DATA EXTRACTION ##########

# 1
def merge_embeddings_and_reg_data():
    """Extracts age data from the healthy train and val datasets and merges them with the embeddings dataset."""
    logger = logging.getLogger("REG 1")
    logger.info("Merging Age Data of Healthy Patients with Embeddings")

    folder_path = "../data/raw/"
    healthy_train_path = folder_path + "healthy_train.csv"
    healthy_test_path = folder_path + "healthy_test.csv"
    embeddings_path = folder_path + "embeddings_cls.csv"
    output_path_partial = "../data/interim/emb_age_healthy_"

    column_names_final = [f"feature_{i}" for i in range(1025)] + ["eid","age"]

    _, emb_list = merge_data_by_eid(
        data_paths=[healthy_train_path, embeddings_path],
        keepfile_indices=[1], 
        output_path=output_path_partial + "train.csv",
        only_cols=column_names_final)

    merge_data_by_eid(
        data_paths=[healthy_test_path],
        loaded_data= emb_list,
        output_path=output_path_partial + "test.csv",
        only_cols=column_names_final)

    logger.info("Age Data: DONE Age data extraction")

# 2
def merge_radiomics_and_embeddings_reg(separate_types=False):
    """Merges the radiomics data with the age+embeddings data."""
    logger = logging.getLogger("REG 2")
    logger.info("Starting combining radiomics and age data of healthy patients")

    extract_radiomics_data(eid_type="healthy", eid_paths=["../data/raw/healthy_train.csv", "../data/raw/healthy_test.csv"])  

    rad_types = ["wat", "fat"]
    set_types = ["train", "test"]
    radiomics_path_partial = "../data/raw/radiomics_healthy_"
    input_path_partial = "../data/interim/emb_age_healthy_"
    output_path_partial = "../data/interim/rad_emb_age_healthy_"
    
    if separate_types:
        for rad_type in rad_types:
            logger.info(f"Loading radiomics {rad_type} data")
            radiomics = pd.read_csv(f"{radiomics_path_partial}{rad_type}.csv")
            for set_type in set_types:
                logger.info(f"Loading age data for {set_type}")
                mae_age_data = pd.read_csv(f"{input_path_partial}{set_type}.csv", usecols=["eid", "age"])
                logger.info(f"Merging radiomics {rad_type} data with age {set_type} data")
                mae_age_data = radiomics.merge(mae_age_data, on="eid", how="inner", suffixes=("", "_y"))
                mae_age_data = mae_age_data.loc[:, ~mae_age_data.columns.str.endswith("_y")]
                mae_age_data.to_csv(f"{output_path_partial}{rad_type}_{set_type}.csv", index=False)
                logger.info(f"Saved radiomics {rad_type} data with age {set_type} data; size: {mae_age_data.shape}")
    else:
        logger.info(f"Loading radiomics data")
        radiomics_fat = pd.read_csv(f"{radiomics_path_partial}fat.csv")
        radiomics_wat = pd.read_csv(f"{radiomics_path_partial}wat.csv")
        # add type to columns
        radiomics_fat.rename(columns=lambda x: f"{x}_fat" if x not in ["eid", "age"] else x, inplace=True)
        radiomics_wat.rename(columns=lambda x: f"{x}_wat" if x not in ["eid", "age"] else x, inplace=True)

        for set_type in set_types:
            logger.info(f"Loading age data for {set_type}")
            mae_age_data = pd.read_csv(f"{input_path_partial}{set_type}.csv", usecols=["eid", "age"])
            logger.info(f"Merging radiomics data with age {set_type} data")
            mae_age_data = radiomics_fat.merge(mae_age_data, on="eid", how="inner", suffixes=("", "_y"))
            mae_age_data = mae_age_data.merge(radiomics_wat, on="eid", how="inner", suffixes=("", "_y"))
            mae_age_data = mae_age_data.loc[:, ~mae_age_data.columns.str.endswith("_y")]
            mae_age_data.to_csv(f"{output_path_partial}ALL_{set_type}.csv", index=False)
            logger.info(f"Saved radiomics data with age {set_type} data; size: {mae_age_data.shape}")

    logger.info("DONE Merging radiomics and age data")

# 3
def create_regression_data(separate_types=False):
    logger = logging.getLogger("REG 3")
    logger.info("Creating regression data files")
    rad_types = ["wat", "fat"] if separate_types else ["ALL"]
    set_types = ["train", "test"]

    input_path_partial = "../data/interim/rad_emb_age_healthy_"
    output_path_partial = "../data/regression/"

    na_cols = [] # keep the same na columns for all datasets for consistency
    for rad_type in rad_types:
        for set_type in set_types:
            logger.info(f"\tLoading {set_type} data for {rad_type}")
            data = pd.read_csv(f"{input_path_partial}{rad_type}_{set_type}.csv")
            logger.info(f"\tData loaded, cleaning data")
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
            logger.info(f"\tSaving {set_type} data for {rad_type}")
            data.to_csv(f"{output_path_partial}{rad_type}_{set_type}.csv", index=False)
    logger.info("Reg: DONE Creating regression data files")

########## FUNCTIONS FOR CLASSIFICATION DATA EXTRACTION ##########

DISEASE_TO_PATH = {
        "cancer": "../data/raw/cancer_timerange_5year.csv",
        "copd": "../data/raw/copd_timerange_5year.csv",
        "liver": "../data/raw/liver_disease_timerange_5year.csv",
        "pancreatic": "../data/raw/pancreatic_disease_timerange_5year.csv",
        "cancer3": "../data/raw/cancer_timerange_3year.csv",
        "cancer4": "../data/raw/cancer_timerange_4year.csv",
        }

# 1
def merge_embeddings_and_class_data(disease_type="cancer"):
    """Merges the embeddings data with the age data."""
    logger = logging.getLogger("CLASS 1")
    logger.info("Starting combining embeddings and time to event data of cancer patients")
    
    
    path = DISEASE_TO_PATH[disease_type]
    embeddings_path = "../data/raw/embeddings_cls.csv"
    output_path = f"../data/interim/emb_{disease_type}.csv"  

    cancer_data = pd.read_csv(path, usecols=["eid"])
    embeddings = pd.read_csv(embeddings_path)
    logger.info("\tMerging embeddings and cancer data")
    cancer_data = cancer_data.merge(embeddings, on="eid", how="inner", suffixes=("", "_y"))
    cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.endswith("_y")]
    logger.info("\tSaving merged data")
    cancer_data.to_csv(output_path, index=False)

# 2
def merge_radiomics_and_embeddings_class(separate_types=False, disease_type="cancer"):
    """Merges the radiomics data with the embeddings and time to event data."""
    logger = logging.getLogger("CLASS 2")
    logger.info("CLASS 2: Starting combining radiomics, embeddings and time to event data of cancer patients")

    extract_radiomics_data(eid_type=disease_type, eid_paths=DISEASE_TO_PATH[disease_type])

    disease_emb_path = DISEASE_TO_PATH[disease_type]
    radiomics_path_partial = f"../data/raw/radiomics_{disease_type}_"
    output_path = f"../data/interim/rad_emb_{disease_type}.csv"

    if separate_types:
        rad_type = "fat"
        logger.info(f"\tLoading radiomics {rad_type} data")
        radiomics = pd.read_csv(f"{radiomics_path_partial}{rad_type}.csv")
        logger.info("\tLoading embeddings and time to event data")
        cancer_emb_data = pd.read_csv(disease_emb_path)
        logger.info("\tMerging radiomics and cancer data")
        cancer_data = radiomics.merge(cancer_emb_data, on="eid", how="inner", suffixes=("", "_y"))
        cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.endswith("_y")]
        logger.info("\tSaving merged data")
        cancer_data.to_csv(output_path, index=False)
    else:
        logger.info(f"\tLoading radiomics data")
        radiomics_fat = pd.read_csv(f"{radiomics_path_partial}fat.csv")
        radiomics_wat = pd.read_csv(f"{radiomics_path_partial}wat.csv")
        # add type to columns
        radiomics_fat.rename(columns=lambda x: f"{x}_fat" if x not in ["eid"] else x, inplace=True)
        radiomics_wat.rename(columns=lambda x: f"{x}_wat" if x not in ["eid"] else x, inplace=True)

        logger.info("\tLoading embeddings and time to event data")
        cancer_emb_data = pd.read_csv(disease_emb_path)
        logger.info("\tMerging radiomics and cancer data")
        cancer_data = radiomics_fat.merge(cancer_emb_data, on="eid", how="inner", suffixes=("", "_y"))
        cancer_data = cancer_data.merge(radiomics_wat, on="eid", how="inner", suffixes=("", "_y"))
        cancer_data = cancer_data.loc[:, ~cancer_data.columns.str.endswith("_y")]
        logger.info("\tSaving merged data")
        cancer_data.to_csv(output_path, index=False)

# 3 WARNING: This requires merge_radiomics_and_embeddings_reg to be run 
def combine_rad_emb_healthy_and_disease(separate_types=False, disease_type="cancer"):
    """Combines the healthy and cancer data for classification."""
    logger = logging.getLogger("CLASS 3")
    logger.info("CLASS 3: Combining healthy and cancer data for classification")
    logger.info(f"\tLoading healthy data")

    healthy_path_partial = "../data/interim/rad_emb_age_healthy_"
    disease_path = f"../data/interim/rad_emb_{disease_type}.csv"
    output_path = f"../data/interim/rad_emb_combined_{disease_type}_{'fat' if separate_types else 'ALL'}.csv"

    if separate_types:
        healthy_train = pd.read_csv(f"{healthy_path_partial}fat_train.csv")
        healthy_test = pd.read_csv(f"{healthy_path_partial}fat_test.csv")
    else:
        healthy_train = pd.read_csv(f"{healthy_path_partial}ALL_train.csv")
        healthy_test = pd.read_csv(f"{healthy_path_partial}ALL_test.csv")
    healthy_combined = pd.concat([healthy_train, healthy_test], axis=0)
    healthy_combined["target"] = 0  # healthy patients are labeled as 0
    del healthy_train, healthy_test # free memory
    logger.info(f"\tLoading cancer data")
    cancer_data = pd.read_csv(disease_path)
    cancer_data["target"] = 1  # cancer patients are labeled as 1
    logger.info("\tCombining healthy and cancer data")
    combined_data = pd.concat([healthy_combined, cancer_data], axis=0)
    logger.info("\tSaving combined data")
    combined_data.to_csv(output_path, index=False)


# 4
def create_classification_data(separate_types=False, disease_type="cancer"):
    """Creates the classification data files."""
    logger = logging.getLogger("CLASS 4")
    logger.info("\tLoading combined data for classification")

    data_path = f"../data/interim/rad_emb_combined_{disease_type}_{'fat' if separate_types else 'ALL'}.csv"
    output_path_partial = f"../data/classification/{disease_type}/{'fat' if separate_types else 'ALL'}_"

    combined_data = pd.read_csv(data_path)
    logger.info("\tCleaning data")
    n = 0.3 # threshold for missing values columns
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
    logger.info("\tSplitting data into train and test sets")
    train_data, test_data = train_test_split(combined_data, test_size=0.2, stratify=combined_data['target'], random_state=42, shuffle=True)
    logger.info("\tSaving train and test data")
    train_data.to_csv(f"{output_path_partial}train.csv", index=False)
    test_data.to_csv(f"{output_path_partial}test.csv", index=False)

### Function for pca

def perform_pca(data_path_train, data_path_val, output_path, ignore_cols=None, n_components=0.95):
    """Performs PCA on the training and validation data and saves the transformed data."""
    logger = logging.getLogger("PCA")
    logger.info("Starting PCA on data")
    logger.info(f"\tLoading training data")
    data = pd.read_csv(data_path_train)
    if ignore_cols is not None:
        target_data = data[ignore_cols]
        data = data.drop(columns=ignore_cols)
    pca = PCA(n_components=n_components)
    logger.info(f"\tFitting PCA on training data")
    pca.fit(data)
    pca_data_train = pca.transform(data)
    pca_data_train = pd.DataFrame(pca_data_train, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    if ignore_cols is not None:
        pca_data_train[ignore_cols] = target_data
    pca_data_train.to_csv(output_path + "pca_train.csv", index=False)
    logger.info(f"\tPCA applied, total features: {pca_data_train.shape[1]-len(ignore_cols)}")
    data = pd.read_csv(data_path_val)
    if ignore_cols is not None:
        target_data = data[ignore_cols]
        data = data.drop(columns=ignore_cols)
    logger.info(f"\tTransforming validation data with PCA")
    pca_data_val = pca.transform(data)
    pca_data_val = pd.DataFrame(pca_data_val, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    if ignore_cols is not None:
        pca_data_val[ignore_cols] = target_data
    pca_data_val.to_csv(output_path + "pca_test.csv", index=False)
    logger.info(f"\tPCA applied, total features: {pca_data_val.shape[1]-len(ignore_cols)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract or change data")

    parser.add_argument("--separate_types", action="store_true", help="Separate radiomics data by type (fat and wat) for regression and classification")
    parser.add_argument("--run_radiomics_all", action="store_true", help="Run radiomics data extraction for all disease types and fat and wat")

    # arguments for regression data extraction
    parser.add_argument("--run_reg", action="store_true", help="Run regression data extraction")
    parser.add_argument("--emb_age_reg", action="store_true", help="Extract embeddings and age data")
    parser.add_argument("--rad_emb_reg", action="store_true", help="Merge radiomics and age data")
    parser.add_argument("--create_reg", action="store_true", help="Create regression data files from radiomics and embeddings data")
    parser.add_argument("--pca_reg", action="store_true", help="Perform PCA on regression data")
    
    # arguments for classification data extraction
    parser.add_argument("--disease_type", type=str, default="cancer", choices=["cancer", "copd", "liver", "pancreatic", "cancer3", "cancer4"], help="Type of disease for classification data extraction")
    parser.add_argument("--run_class", action="store_true", help="Run classification data extraction")
    parser.add_argument("--emb_tte_class", action="store_true", help="Extract embeddings and time to event data for classification")
    parser.add_argument("--rad_emb_class", action="store_true", help="Merge radiomics and embeddings data for classification")
    parser.add_argument("--combine_rad_emb_healthy_and_disease", action="store_true", help="Combine healthy and disease data for classification")
    parser.add_argument("--create_class", action="store_true", help="Create classification data files from combined data")
    parser.add_argument("--pca_class", action="store_true", help="Perform PCA on classification data")
    
    parser.add_argument("--test_data_mgr", action="store_true", help="Run the data manager test")
    
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)8s - %(name)10s - %(message)s', filename='logs/data_extraction.log', filemode='w')

    logging.info("###### STARTING DATA EXTRACTION #######")
    total_operations = sum([arg for arg in vars(args).values() if isinstance(arg, bool) and arg])
    if args.run_reg:
        total_operations += 3
    if args.run_class:
        total_operations += 4
    logging.info(f"#### Total operations to run: {total_operations}")

    disease_type = args.disease_type if args.disease_type else "cancer"
    radiomics_type = "fat" if args.separate_types else "ALL"
    regression_path = f"../data/regression/{radiomics_type}"
    classification_path = f"../data/classification/{disease_type}/{radiomics_type}_"

    if args.run_radiomics_all:
        logging.info("Running radiomics data extraction for all disease types")
        extract_radiomics_data(eid_type="healthy", eid_paths=["../data/raw/healthy_train.csv", "../data/raw/healthy_test.csv"])
        for disease, path in DISEASE_TO_PATH.items():
            if disease == "cancer3" or disease == "cancer4":
                continue
            logging.info(f"Extracting radiomics data for {disease} from {path}")
            extract_radiomics_data(eid_type=disease, eid_paths=[path])
        

    if args.test_data_mgr:
        d_logger = logging.getLogger("D MANAGER TEST")
        d_logger.info("Running data manager test")
        dm = DataManager("classification", "liver", "emb", "rboth", "pca", logger=d_logger)
        train_data = dm.get_train()
        d_logger.info(f"Train data shape: {train_data.shape}")
        d_logger.info(train_data.head(3))
        d_logger.info("Data manager test completed successfully")
        sys.exit(0)

    if args.run_reg:
        args.emb_age_reg = True
        args.rad_emb_reg = True
        args.create_reg = True
        args.pca_reg = True

    if args.emb_age_reg:
        try:
            merge_embeddings_and_reg_data()
        except Exception as e:
            # log whole error and exit
            logging.exception(f"Error occurred while merging embeddings and regression data: {e}" )
            sys.exit(1)
    if args.rad_emb_reg:
        try:
            merge_radiomics_and_embeddings_reg(separate_types=args.separate_types)
        except Exception as e:
            logging.exception(f"Error occurred while merging radiomics and regression data: {e}")
            sys.exit(1)
    if args.create_reg:
        try:
            create_regression_data(separate_types=args.separate_types)
        except Exception as e:
            logging.exception(f"Error occurred while creating regression data: {e}")
            sys.exit(1)
    if args.pca_reg:
        try:
            perform_pca(f"{regression_path}/{radiomics_type}_train.csv", f"{regression_path}/{radiomics_type}_test.csv", f"{regression_path}/{radiomics_type}_", ignore_cols=["eid", "age"], n_components=0.95)
        except Exception as e:
            logging.exception(f"Error occurred while performing PCA on regression data: {e}")
            sys.exit(1)

    if args.run_class:
        args.emb_tte_class = True
        args.rad_emb_class = True
        args.combine_rad_emb_healthy_and_disease = True
        args.create_class = True
        args.pca_class = True
    if args.emb_tte_class:
        try:
            merge_embeddings_and_class_data(disease_type=disease_type)
        except Exception as e:
            logging.exception(f"Error occurred while merging embeddings and classification data: {e}")
            sys.exit(1)
    if args.rad_emb_class:
        try:
            merge_radiomics_and_embeddings_class(separate_types=args.separate_types, disease_type=disease_type)
        except Exception as e:
            logging.exception(f"Error occurred while merging radiomics and classification data: {e}")
            sys.exit(1)
    if args.combine_rad_emb_healthy_and_disease:
        try:
            combine_rad_emb_healthy_and_disease(separate_types=args.separate_types, disease_type=disease_type)
        except Exception as e:
            logging.exception(f"Error occurred while combining healthy and disease data: {e}")
            sys.exit(1)
    if args.create_class:
        try:
            create_classification_data(separate_types=args.separate_types, disease_type=disease_type)
        except Exception as e:
            logging.exception(f"Error occurred while creating classification data: {e}")
            sys.exit(1)
    if args.pca_class:
        try:
            perform_pca(f"{classification_path}_train.csv", f"{classification_path}_test.csv", f"{classification_path}_", ignore_cols=["eid", "target"], n_components=0.95)
        except Exception as e:
            logging.exception(f"Error occurred while performing PCA on classification data: {e}")
            sys.exit(1)

    logging.info("DONE ALL")

