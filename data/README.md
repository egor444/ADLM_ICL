# This folder: Data

## regression
Data as inputs for the age prediction regression task                
- train_wat/fat:                   training sets with normalized MAE embeddings, radionomic features water/fat respectively and age (and eids)
- test_wat/fat:                    test sets as above, but smaller
- train_wat/fat_pca:               data after PCA with only pca inputs and age
- test_wat/fat_pca:                the pca transformed test data

## classification
Data as inputs for the risk assessment binary calssification task

## other:
- old_tabular_mae_healthy_train/healthy_val:        MAE embedding data + other tabular data + age
- pca_components_radiomics_wat:     665 PCs from water radiomics + age
- radiomics_embeddings_fat_top500:  top 500 correlating features to age from rad & emb + age
- radiomics_embeddings_fat/wat:     all radiomics + embeddings + age
- radiomics_fat/wat:                just radiomics and eids collected (incomplete)
- radiomics_fat/wat_v2:             just radiomics and eids collected (incomplete, more stuff)
- tte_embeddings:                   embeddings and time to event data for tte < 5 years
