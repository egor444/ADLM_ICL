# Data Extraction

## Structure
This folder contains files to extract and combine the nessesary input data. \
- ```data_extraction.py``` contains old functions to extarct data and recombine it in the data folder of the project. It is also responsible for getting the nessesary radiomics files for the reqired EIDs and combininffg them into one file.
- ```data_manager.py``` contains a new DataManager class, which when imported handles all the data loading and combining based on input tags.

## DataManager
### **Examples on how to use the DataManager**

```py
datamanager = DataManager("regression", "emb", "pca")
``` 
This creates a DataManager for an age prediction regression task while using embeddings and applying dimentionality reduction via PCA on the data. \
The training and testing datasets can now be gotten in the form of a pd.DataFrame with:

```py
train_set = datamanager.get_train()
test_set = datamanager.get_train()
``` 

**OR** for five fold cross-validation a subset of the training set can be retreived by passing a list of all the indices of folds, like 
```py
fold_subset = datamanger.get_fold_data_set([0,1,3,4])
``` 
Which will return one DataFrame with the data of all folds, except fold 2.

### Second Example
```py
dm_logger = logging.getLogger('DataManager')
datamanager = DataManager("classification", "liver", "emb", "rboth", "force", "verbose", logger=dm_logger)
```

Loads data for classififcation on healthy/liver problems with MAE embeddings, radiomics for both water and fat, forces the recreation of the data regardless if it exists and loggs the process in the custom logger. \
Now we can get the CV sets for training and validation: 

```py
fold_train_set = datamanger.get_fold_data_set([0,1,2,3])
fold_val_set = datamanger.get_fold_data_set([4])
```


### **DataManager Flags**
- *Task:* ```regression```, ```classification```
- *MAE embeddings:* ```emb```
- *Radiomic data:* ```rfat```, ```rwat```, ```rboth```
- *Disease types for classification:* ```cancer```, ```copd```, ```liver```, ```pancreatic```
- *Apply PCA:* ```pca```
- *Dont save the data combination in the data folder:* ```nosave```
- *Dont load the data if it already exists and recombine it from scratch:* ```force```
- *Log output:* ```verbose```

### **DataManager Parameters**
- *Path to data folder, if dafault is incorrect:* ```data_folder_path```
- *Data test/train split (default: 0.2):* ```split```
- *Pass a logger, otherwise default output is print (Reqires verbose flag):* ```logger```
- *Number of folds on thje train set (default 5):* ```k_folds```


