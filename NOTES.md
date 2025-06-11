# Project Notes

### Notes to discuss in meeting on 21.05.
- define goal clearly, what is our measure of success?
- what models, methods to use
- what are the inputs (data fields, information)
- define next steps -> First approaches, encountered issues 
- what is the work split


### Notes from Meeting 07.05.
- Use MAE encodings for predictions (path below)
- Try BMI prediction

### Personal Notes (Egor)
- Analyse embedding to age correlations, extract most important embeddings for efficiency
- Same for other tasks (BMI, Cancer binary class..)
- Ideas for baseline models: Simple MLP or CNN (as very simple feature -> age predictor), XGBoost

### Paths (From Dima's google docs)
| Descr | Location |
| --- | --- |
|Project directory: | /vol/miltank/projects/ukbb/ |
| Csv file with all fields | /vol/miltank/projects/ukbb/677795/ukb677795.csv |
| For convenient access, use the phenos as well | /vol/miltank/projects/ukbb/677795/phenos |
| Where are the original files? | /vol/miltank/projects/ukbb/data/whole_body/nifti |
| Where are all masks from the total_segmenator? | /vol/miltank/projects/ukbb/data/whole_body/total_segmentator/ |
| Whew are all radimics | /vol/miltank/projects/ukbb/data/whole_body/radiomics \
| Labels for classification and regression are under | /vol/miltank/projects/ukbb/projects/practical_ss25_icl/ |
| Embeddings | /vol/miltank/projects/ukbb/data/whole_body/mae_embeddings/ |

Use ITK Viewer or 3D Slicer for visualization of 3D niftis

### Links
https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/first_occurrences_outcomes.pdf
https://biobank.ndph.ox.ac.uk/showcase/ukb/docs/HospitalEpisodeStatistics.pdf#page3
https://biobank.ndph.ox.ac.uk/showcase/refer.cgi?id=141140

[MAE repo](https://github.com/yayapa/WholeBodyRL) \
[Imbalanced regression repo](https://github.com/ismailnejjar/IM-Context) \
[TabPFN repo](https://github.com/PriorLabs/TabPFN)


