# In-Context Learning for Image Derived Features in The Medical Domain

## Our goal:
Establish in-context learning as a viable methodology for working with mdeical images and their derrived features.

We try to apply In-Context learning methods, namely **TabPFN** and **GPT-2** compare them to a selctvariety of selected baseline models on two distinct tasks.

| Regression | Classification |
| --- | --- |
| Try to predict the biological age of healthy ppeople based on our given tabular data. | Try to predict weather an individual is at risk for a certain category of disease in the next 5-year window. | 

The raw data we use are Masked Autoencoder Embeddings from MRI images and extracted radiomic features with water and fat contrast from the images.

For disease risk assesement we performed binary classififcation (not at-risk/at-risk) on the following classes:
- Cancer types (various)
- Liver diseases
- Pancreatic diseases
- Chronic Obstructive Pulmonary Disease

## Project Structure

| Folder Name | Description |
| --- | --- |
|**data** | Raw and preprocessed data files, more info in the ```data/README.md``` |
|**data_analysis** | Notebooks for looking into and analysing data and plotting simple plots |
|**data_extraction** | Combination, creation and processing of data to use as input |
|**classification** | Model training an testing for classifications tasks|
|**regression** | Model training an testing for regression tasks |
|**feature_engineering** | Classes and functions for feature engineering, as well as finetuning |

## How to Use

1. Copy the supplied data folder into the project directory. (Just replace the data folder with the README inside with the data folder, that includes the data)
2. Navigate in the folder for the task you wish to run, either **regression** or **classification**

