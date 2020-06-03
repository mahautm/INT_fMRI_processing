#INT_fMRI_processing

feature_extraction_ABIDE prepares data from a given set of subjects in the ABIDE data set, and prepares them for feature extraction through an auto-encoder by producing an activation matrix, and one representative of anatomical girification.

## Functions
As represented in the flow chart, the steps are as follows :
download_abide-urls : downloads all required data for chosen subjects
register : sets

## requires :

### software :

FSL
Freesurfer
Matlab Runtime

### other :

Freesurfer's SUBJECTS_DIR environment variable must be made to correspond to where subject data is downloaded (default is /scratch/mmahaut/processed_abide)
the template files to be used (default is fsaverage5) must be in the freesurfer SUBJECTS_DIR

## To run :

Prepare the subs_list JSON file in url_preparation to contain the list of subjects you wish to prepare matrices for.

### Then, you have to options :

1. run the script from the mesocentre, with SLURM : in this case, call mesocentre_ABIDE, with python3, each subject will be a separate job

2. run from a single machine : use feature_extraction_ABIDE's extract_all_ABIDE function to extract features on each subject, one after the other.
