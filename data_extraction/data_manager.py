
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import logging
from sklearn.preprocessing import StandardScaler
from PIL import Image


################################################################################
# DataManager Class
#
# This class manages the data for different tasks, diseases, and feature types.
# All possible flags and arguments are defined in the class below.
################################################################################


PATH_DICT = {
    'embeddings': '/vol/miltank/projects/ukbb/data/whole_body/mae_embeddings/embeddings_cls.csv',
    'healthy_train': '/vol/miltank/projects/ukbb/projects/practical_ss25_icl/whole_body_3d_healthy_noselfreported_noicd10_assessment2_train_df.csv',
    'healthy_test': '/vol/miltank/projects/ukbb/projects/practical_ss25_icl/whole_body_3d_healthy_noselfreported_noicd10_assessment2_val_df.csv',
    'data_default': '/vol/miltank/projects/practical_sose25/in_context_learning/data',
    'img_folder': '/vol/miltank/projects/ukbb/data/whole_body/nifti_2d'
}

DISEASE_TO_PATH = {
    'cancer': 'cancer_timerange_5year.csv',
    'copd' : 'copd_timerange_5year.csv',
    'liver': 'liver_disease_timerange_5year.csv',
    'pancreatic': 'pancreatic_disease_timerange_5year.csv',
    'cancer4': 'cancer_timerange_4year.csv',
    'cancer3': 'cancer_timerange_3year.csv',
}

RAD_TYPES = ['rnone','rfat','rwat','rboth']
ALL_FLAGS = ['classification', 'regression', 'emb', 'img', 'force', 'nosave', 'pca'] + list(DISEASE_TO_PATH.keys()) + RAD_TYPES
FILE_FLAGS = ['emb', 'img', 'rfat','rwat','rboth']
UNCHANGED_COLS = ['eid', 'target', 'age', 'img']  # Columns that should not be changed during processing

class DataManager:
    '''
    Class for managing all combinations of datasets. Generates a dataset, splitting it if nessessary, and saving it to the specified path.
    If the data already exists, it will load the data from the specified path.

    Possible flags:
    - classification: for classification tasks
    - regression: for regression tasks

    - <disease>: specify disease type for classification tasks (e.g. 'cancer', 'copd', 'liver', 'pancreatic')
    - emb: use embeddings (default: False)
    - img: use images (default: False)

    - rfat: use radiomics features for fat (default: False)
    - rwat: use radiomics features for water (default: False)
    - rboth: use both radiomics features for both fat and water (default: False)

    - nosave: do not save the data (default: False)
    - force: force recombining of data even if it already exists in the data folder (default: False)
    - verbose: print output of the data manager (default: False)

    - pca: apply PCA transformation to the data (default: False)

    Possible arguments:
    - data_folder_path: path to the data folder
    - split: fraction of data to be used for testing (default: 0.2)
    - logger: a logger object to log messages (default: None, prints to console)

    Example usage:
    dm = DataManager('classification', 'cancer', 'emb', 'rboth', logger=my_logger)
    train_data = dm.get_train()
    test_data = dm.get_test()
    This will initialize the DataManager for a classification task on cancer with embeddings and both radiomics.
    '''

    def __init__(self, *flags, **kvargs):
        ###### Initialize data manager with flag types #######
        self.logger = kvargs.get('logger', None)
        # Base flags
        self.task = 'classification' if 'classification' in flags else 'regression'
        self.save_data = False if 'nosave' in flags else True
        if not 'classification' in flags and not 'regression' in flags:
            self.log("WARNING: No task specified, defaulting to regression.", level=logging.WARNING)
        self.disease = next(iter(set(flags) & set(DISEASE_TO_PATH.keys())), None)
        if self.task == 'regression' and self.disease is not None:
            self.log("WARNING: Disease specified for regression task, setting to classification.", level=logging.WARNING)
            self.task = 'classification'
        self.verbose = True if 'verbose' in flags else False
        # Data folder paths
        self.data_folder_path = kvargs.get('data_folder_path', PATH_DICT['data_default'])
        self.interim = self.data_folder_path + '/interim/'
        self.raw = self.data_folder_path + '/raw/'
        self.task_out_path = self.data_folder_path + '/' + self.task + '/' 
        if self.task == 'classification' and self.disease is None:
            raise ValueError("Disease must be specified for classification task. Available disease types: " + ', '.join(DISEASE_TO_PATH.keys()))
        self.out_path = self.task_out_path + self.disease + "/" if self.task == 'classification' else self.task_out_path + "/"
        # select radiomics type
        self.rad_type = next(iter(set(flags) & set(RAD_TYPES)), 'rnone')
        # select embeddings
        self.emb = PATH_DICT['embeddings'] if 'emb' in flags else None
        # images directly
        self.img = True if 'img' in flags else False
        # Log message
        params = ''
        if self.emb:
            params += 'embeddings, '
        if self.img:
            params += 'images, '
        if self.rad_type != 'rnone':
            params += f'radiomics ({self.rad_type}), '
        if self.task == 'classification':
            params += f'disease: {self.disease}, '
        self.log(f"Initializing Data Manager for {self.task} with parameters: " + params)
        unrecognized_flags = [flag for flag in flags if flag not in ALL_FLAGS]
        if unrecognized_flags:
            self.log(f"Warning: Unrecognized flags {unrecognized_flags} will be ignored.", level=logging.WARNING)
        # Initialize output paths
        self.file_flags = [flag for flag in FILE_FLAGS if flag in flags]
        flagstring = '_'.join(self.file_flags)
        flagstring = flagstring + "_" if flagstring else ''
        self.outfile_paths = [
            self.out_path + flagstring + 'train.csv',
            self.out_path + flagstring + 'test.csv'
        ]
        # check if output paths exist
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        # check if output files already exist
        if os.path.exists(self.outfile_paths[0]) and os.path.exists(self.outfile_paths[1]) and not 'force' in flags:
            self.log(f"Data already exists. Loading data.", level=logging.INFO)
            self.data = [pd.read_csv(self.outfile_paths[0]), pd.read_csv(self.outfile_paths[1])]
            if 'pca' in flags:
                self.apply_pca(n_components=kvargs.get('n_components', 0.95))
            return
        self.data = []
        self.init_eids(split=kvargs.get('split', 0.2))
        self.create_data()
        if 'pca' in flags:
            self.apply_pca(n_components=kvargs.get('n_components', 0.95))

    def get_train(self):
        ''' Get the training data '''
        if not self.data:
            raise ValueError("Data not initialized.")
        return self.data[0].drop(columns=['eid'])

    def get_test(self):
        ''' Get the test data '''
        if not self.data:
            raise ValueError("Data not initialized.")
        return self.data[1].drop(columns=['eid'])

    def load_from_path(self, path):
        ''' Load a dataframe from the specified path'''
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        # self.log(f"Loading data from {path}")
        df = pd.read_csv(path)
        return df

    def init_eids(self, split = 0.2):
        ''' Initialize eids and targets for the dataset '''
        h_train = self.load_from_path(PATH_DICT['healthy_train'])
        h_train = h_train[['eid', 'age']]
        h_test = self.load_from_path(PATH_DICT['healthy_test'])
        h_test = h_test[['eid', 'age']]
        if self.task == 'classification':
            self.log("Initializing eids for classification task.", level=logging.INFO)
            # Combine all healthy eids and assign target 0
            h_total = pd.concat([h_train, h_test], ignore_index=True)
            h_total = h_total.drop(columns=['age'])
            h_total['target'] = 0  # Healthy individuals are labeled as 0
            disease_eids = self.load_from_path(self.raw + DISEASE_TO_PATH[self.disease])
            disease_eids = disease_eids[['eid']]
            disease_eids['target'] = 1
            # Combine healthy and disease eids
            eids = pd.concat([h_total, disease_eids], ignore_index=True)
            eids = eids.sample(frac=1) # Shuffle the eids
            train, test = train_test_split(eids, test_size=split, stratify=eids['target'], shuffle=True)
            self.data = [train, test]
        elif self.task == 'regression':
            self.log("Initializing eids for regression task.", level=logging.INFO)
            self.data = [h_train, h_test]
        self.log(f"Initialized eids with shapes: train {self.data[0].shape}, test {self.data[1].shape}", level=logging.INFO)
    
    def combine_data(self, new_data):
        ''' Combine new data columns with existing data by eid '''
        if not self.data:
            raise ValueError("Data not initialized. Call init_eids() first.")
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("new_data must be a pandas DataFrame.")
        if 'eid' not in new_data.columns:
            raise ValueError("The new data must contain 'eid' column.")
        self.log(f"Combining data with shape {new_data.shape} with dataset.", level=logging.INFO)
        new_train = pd.merge(self.data[0], new_data, on='eid', how='left')
        new_test = pd.merge(self.data[1], new_data, on='eid', how='left')
        self.data = [new_train, new_test]
        self.log(f"Data combined. New shapes: train {self.data[0].shape}, test {self.data[1].shape}", level=logging.INFO)

    def load_images(self, eids, path = PATH_DICT['img_folder']):
        ''' Load 2d images from the specified path '''
        images = []
        out_eids = [eid for eid in eids if os.path.exists(os.path.join(path, f"{eid}.png"))]
        if len(eids) - len(out_eids) > 0:
            self.log(f"WARNING: {len(eids) - len(out_eids)} eids do not have corresponding images.", level=logging.WARNING)
        for eid in out_eids:
            img_path = os.path.join(path, f"{eid}.png")
            img = Image.open(img_path).getdata()
            # convert to list
            img = list(img)
            images.append(img)

        # image dataframe with one column 'img'
        img_frame = pd.DataFrame({'img': images, 'eid': out_eids})
        return img_frame

    def create_data(self):
        ''' Create the final data by combining all data sources '''
        if not self.data:
            raise ValueError("Data not initialized. Call init_eids() first.")
        if self.emb:
            self.log(f"Loading embeddings from {self.emb}", level=logging.INFO)
            emb_data = self.load_from_path(self.emb)
            emb_cols = [f"feature_{i}" for i in range(1025)] + ["eid","age"]
            emb_data = emb_data[emb_cols]
            self.combine_data(emb_data)
        if self.img:
            self.log("Loading images.", level=logging.INFO)
            all_eids = self.data[0]['eid'].tolist() + self.data[1]['eid'].tolist()
            images = self.load_images(all_eids)
            self.combine_data(images)
        if self.rad_type != 'rnone':
            # Fat Radiomics
            if self.rad_type == 'rfat' or self.rad_type == 'rboth':
                self.log("Loading radiomics features for fat.", level=logging.INFO)
                rad_healthy_fat = self.load_from_path(self.raw + f'radiomics/radiomics_healthy_fat.csv')
                if self.task == "classification":
                    rfat_data = self.load_from_path(self.raw + f'radiomics/radiomics_{self.disease}_fat.csv')
                    if self.rad_type == 'rfat':
                        rfat_data = pd.concat([rad_healthy_fat, rfat_data], ignore_index=True)
                        self.combine_data(rfat_data)
                if self.task == "regression" and self.rad_type == 'rfat':
                    self.combine_data(rad_healthy_fat)
                
            # Water Radiomics
            if self.rad_type == 'rwat' or self.rad_type == 'rboth':
                self.log("Loading radiomics features for water.", level=logging.INFO)
                rad_healthy_wat = self.load_from_path(self.raw + f'radiomics/radiomics_healthy_wat.csv')
                if self.task == 'classification':
                    rwat_data = self.load_from_path(self.raw + f'radiomics/radiomics_{self.disease}_wat.csv')
                    if self.rad_type == 'rwat':
                        rwat_data = pd.concat([rad_healthy_wat, rwat_data], ignore_index=True)
                        self.combine_data(rwat_data)
                if self.task == 'regression' and self.rad_type == 'rwat':
                    self.combine_data(rad_healthy_wat)
            # Both Radiomics
            if self.rad_type == 'rboth':
                self.log("Combining radiomics features for fat and water.", level=logging.INFO)
                rad_healthy_fat = rad_healthy_fat.rename(columns=lambda x: x + '_fat' if x != 'eid' else x)
                rad_healthy_wat = rad_healthy_wat.rename(columns=lambda x: x + '_wat' if x != 'eid' else x)
                combined_rad_healthy = pd.merge(rad_healthy_fat, rad_healthy_wat, on='eid', how='left')
                if self.task == 'regression':
                    self.combine_data(combined_rad_healthy)
                else:
                    rfat_data = rfat_data.rename(columns=lambda x: x + '_fat' if x != 'eid' else x)
                    rwat_data = rwat_data.rename(columns=lambda x: x + '_wat' if x != 'eid' else x)
                    combined_rad_data = pd.merge(rfat_data, rwat_data, on='eid', how='left')
                    combined_all = pd.concat([combined_rad_healthy, combined_rad_data], ignore_index=True)
                    self.combine_data(combined_all)
        # clean data
        # drop columns with non numeric values
        drops_cols = [col for col in self.data[0].columns if col not in UNCHANGED_COLS]
        exclude_cols = self.data[0][drops_cols].select_dtypes(exclude=[np.float64, np.int64, np.number]).columns.tolist()
        self.data[0] = self.data[0].drop(columns=exclude_cols)
        self.data[1] = self.data[1].drop(columns=exclude_cols)
        # drop columns with more than 30% nan values, except 'eid', 'target', 'age', 'img'
        nanprc = 0.3
        drops_cols = [col for col in self.data[0].columns if col not in UNCHANGED_COLS]
        keep_cols = list(self.data[0][drops_cols].dropna(thresh=int((1 - nanprc) * self.data[0].shape[0]), axis=1).columns)
        keep_cols = [col for col in self.data[0] if col in UNCHANGED_COLS] + keep_cols
        self.data[0] = self.data[0][keep_cols]
        self.data[1] = self.data[1][keep_cols]
        # drop rows with nan values
        self.data[0] = self.data[0].dropna()
        self.data[1] = self.data[1].dropna()
        # normalize data
        scaler = StandardScaler()
        cols_to_normalize = [col for col in self.data[0].columns if col not in UNCHANGED_COLS]
        if len(cols_to_normalize) == 0:
            self.log("No columns to normalize. Skipping.", level=logging.INFO)
        else:
            self.log(f"Normalizing columns: {len(cols_to_normalize)}", level=logging.INFO)
            self.data[0][cols_to_normalize] = scaler.fit_transform(self.data[0][cols_to_normalize])
            self.data[1][cols_to_normalize] = scaler.transform(self.data[1][cols_to_normalize])
        # drop age column if classification task
        if (self.task == 'classification') and ('age' in self.data[0].columns):
            self.data[0] = self.data[0].drop(columns=['age'])
            self.data[1] = self.data[1].drop(columns=['age'])
        # save data if no nosave flag is set
        if self.save_data:
            self.log(f"Saving data to {self.outfile_paths[0]} and {self.outfile_paths[1]}", level=logging.INFO)
            self.data[0].to_csv(self.outfile_paths[0], index=False)
            self.data[1].to_csv(self.outfile_paths[1], index=False)
        self.log(f"Data created with shapes: train {self.data[0].shape}, test {self.data[1].shape}", level=logging.INFO)
    
    def img_to_np(self):
        ''' Convert image column to numpy arrays '''
        if not self.data:
            raise ValueError("Data not initialized. Call init_eids() first.")
        if 'img' not in self.data[0].columns or 'img' not in self.data[1].columns:
            raise ValueError("Image column 'img' not found in data.")
        self.log("Converting image column to numpy arrays.", level=logging.INFO)
        self.data[0]['img'] = self.data[0]['img'].apply(lambda x: np.array(eval(x)))
        self.data[1]['img'] = self.data[1]['img'].apply(lambda x: np.array(eval(x)))

    def apply_pca(self, n_components=0.95, to_img=False):
        ''' Apply PCA transformation to the data '''
        if not self.data:
            raise ValueError("Data not initialized. Call init_eids() first.")
        self.log(f"Applying PCA transformation with {n_components} components.", level=logging.INFO)
        pca = PCA(n_components=n_components)
        # Fit PCA on training data
        pca_cols = [col for col in self.data[0].columns if col not in UNCHANGED_COLS]
        meta_cols = [col for col in self.data[0].columns if col in UNCHANGED_COLS]
        if len(pca_cols) == 0 and not to_img:
            self.log("No columns to apply PCA on. Skipping.", level=logging.INFO)
            return
        if len(pca_cols) > 0:
            train_frame = pd.DataFrame(pca.fit_transform(self.data[0][pca_cols]))
            test_frame = pd.DataFrame(pca.transform(self.data[1][pca_cols]))
            self.data[0] = pd.concat([self.data[0][meta_cols], train_frame], axis=1)
            self.data[1] = pd.concat([self.data[1][meta_cols], test_frame], axis=1)
            self.log(f"PCA applied. Number of components: {train_frame.shape[1]}", level=logging.INFO)
        if to_img:
            if 'img' not in self.data[0].columns or 'img' not in self.data[1].columns:
                raise ValueError("Image column 'img' not found in data.")
            self.data[0]['img'] = list(pca.transform(np.stack(self.data[0]['img'].values)))
            self.data[1]['img'] = list(pca.transform(np.stack(self.data[1]['img'].values)))
            self.log(f"PCA applied to images. Number of components: {len(self.data[0]['img'].iloc[0])}", level=logging.INFO)
        

    def log(self, message, level=logging.INFO):
        if not self.verbose:
            return
        if self.logger:
            self.logger.log(level, message)
        else:
            print(message)
        