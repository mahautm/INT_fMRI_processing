# INT_fMRI_processing
# Part 1 : feature extraction
![flow_chart for feature extraction](https://github.com/mahautm/INT_fMRI_processing/blob/master/documentation/extract_one_ABIDE.png)
feature_extraction_ABIDE prepares data from a given set of subjects in the ABIDE data set, and prepares them for feature extraction through an auto-encoder by producing an activation matrix, and one representative of anatomical girification.

## Functions
As represented in the flow chart, the modules run as follows :

*download_abide-urls* : downloads all required data for chosen subjects

*compute_rsfmri* : computes rsfmri connectivity matrices between regions of interest and brain vertices

*compute_gyrification* : computes a matrix made of 100 eigenvectors, representing the frequency of spatial brain folds, also named gyrifications


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

# Part 2 : learning
## A. multi-modal auto-encoder
The auto-encoders aim to reduce noise in images by producing a smaller latent layer retaining a maximum of information.
All scripts suffixed with "mdae" contribute to that aim.

  1. mdae.py
    will launch one script (mdae_step.py) per fold on different slurm jobs. Different sizes for each modality can be tested.
  2. mdae_conv.py 
    Not yet relevant. Does the same as mdae.py, but with a convolutional model. As the vertex used for training do not retain 2d spatial significance, the laplatian should be used, but has not yet been implemented.
  3. mdae_stats.py
    Once a model has been trained, this script will extract reconstruction MSE, and save it as .npy matrices in relevant folders.
  4. mdae_stats_analysis.py
    will take matrices built by mdae_stats.py and make a simple table out of it, for easier analysis. Ideally, this script should be merged with mdae_stats.py
  5. mdae_step.py
    trains and saves a model on a specific fold, data set, and modality combination.
  6. mdae_step_conv.py 
    same as mdae_step.py, but with a convolutional model
  7. mdae_step_vertices.py
    same as mdae_step.py, but training folds are done by vertex, and not by subject
  8. mdae_vertices.py
    same as mdae.py, but training folds are done by vertex, and not by subject
    
## B. trace-regression
All scripts prefixed with "regression" aim to predict the text score (interTVA) from a subject or classify (ABIDE) subjects.

  1. regression.py
    generates and save the beta matrix for regression.
  2. regression_hyperparam.py
    calls a different regression script for each fold, and each different hyperparameter combination
  3. regression_raw.py
    same as regression, but loads raw data, that has not been through the auto-encoder
  4. regression_stats.py
    gives mse and r squared for chosen modalities and data sources
  5. regressions_stats_hyperparameters.py
    makes and saves regressions scores for all different parameters in a grid, with one table per modality and per data source.
  6. trace_regression.py
    the original unmodified code from Sellami Akrem's code
  7. convexminimization.py
    used by regression to find cost function minimum with fista algorythm
