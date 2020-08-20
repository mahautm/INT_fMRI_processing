# Prepares all the different slurm jobs to run regressions on
# Separate slurm jobs will be made for different folds, different values of delta or of soft_threshold,
# can generate all modalities, raw or not, from all data_sources at once or separately
import os


def run_slurm_job_regression(
    data_orig,
    data_type,
    delta,
    soft_threshold,
    fold,
    email="mmahaut@ensc.fr",
    logs_dir="/scratch/mmahaut/scripts/logs",
    python_path="python",
    slurm_dir="/scratch/mmahaut/scripts/slurm",
    code_dir="/scratch/mmahaut/scripts/INT_fMRI_processing",
    script_name="regression.py",
):
    """
    This will write a .sh file to train a multi-modal auto-encoder on a specific set of subjects with SLURM, and then execute it.

    Parameters
    ----------
    data_orig : {"ABIDE","interTVA"}
        indicates which data set is used. ABIDE is a dataset with subjects on the autism spectrum and control subjects,
        InterTVA is a dataset where non-pathological subjects are given sound recognition tasks. 

    data_type : {"tfMRI","gyrification"}
        The multi-modal auto-encoder uses two modalities to build it's representations. One is resting-state fMRI, and the other
        is either task fMRI (tfMRI) or an anatomical modality which represents the folds in the subject's brain, (gyrification)

    delta : float
        size of laplacian, when looking to get similar results for near areas in the brain.

    soft_threshold : float
        neurones below that value won't activate.

    fold : int
        For cross-validation, subjects are separated in a train and a test group. Those groups must be saved in the corresponding folder
        the number corresponding to each fold will here be used to load the proper group in the mdae-step.py script

    email : string, email address, default "mmahaut@ensc.fr"
        email to which notifications will be sent at the end of the execution of the script for the first fold of a given dimension

    logs_dir : string, path, default "/scratch/mmahaut/scripts/logs"
        path to where both the output and the error logs will be saved.

    python_path : string, path, default "python"
        path to where the python version used to run the mdae_step.py script is found. 3.7 & 3.8 work.
    
    slurm_dir : string, path, default "/scratch/mmahaut/scripts/slurm"
        path to where the .sh script used to enter the slurm queue will be saved before being executed

    code_dir : string, path, default "/scratch/mmahaut/scripts/INT_fMRI_processing"
        path to where the script can be found

    script_name : string, path, default "mdae_step.py"
        script file name
    """

    job_name = "hyperparam_{}_{}_d{}_thres{}".format(
        data_orig, data_type, delta, soft_threshold
    )
    slurmjob_path = os.path.join(slurm_dir, "{}.sh".format(job_name))
    create_slurmjob_cmd = "touch {}".format(slurmjob_path)
    os.system(create_slurmjob_cmd)

    # write arguments into the slurmjob file
    with open(slurmjob_path, "w") as fh:
        fh.writelines("#!/bin/sh\n")
        fh.writelines("#SBATCH --job-name={}\n".format(job_name))
        fh.writelines("#SBATCH -o {}/{}_%j.out\n".format(logs_dir, job_name))
        fh.writelines("#SBATCH -e {}/{}_%j.err\n".format(logs_dir, job_name))
        fh.writelines("#SBATCH --time=24:00:00\n")
        fh.writelines("#SBATCH --account=b125\n")
        fh.writelines("#SBATCH --partition=skylake\n")
        # fh.writelines("#SBATCH --gres-flags=enforce-binding\n")
        # number of nodes for this job
        fh.writelines("#SBATCH --nodes=1\n")
        # number of cores for this job
        fh.writelines("#SBATCH --ntasks-per-node=10\n")
        # email alerts
        fh.writelines("#SBATCH --mail-type=END\n")
        fh.writelines("#SBATCH --mail-user={}\n".format(email))
        # making sure group is ok for data sharing within group
        batch_cmd = (
            'eval "$(/scratch/mmahaut/tools/Anaconda3/bin/conda shell.bash hook)"\n'
            + "conda activate tf\n"
            + "{} {}/{} {} {} {} {} {}".format(
                python_path,
                code_dir,
                script_name,
                data_orig,
                data_type,
                delta,
                soft_threshold,
                fold,
            )
        )
        fh.writelines(batch_cmd)

    os.system("sbatch %s" % slurmjob_path)


if __name__ == "__main__":

    params_grid = {}
    params_grid["delta"] = [0, 1e-2, 1e-3, 1e-4]  # list values to grid_search on
    params_grid["soft_thresh"] = [5e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
    params_grid["modality"] = ["tfMRI"]  # ["gyrification", "tfMRI"]
    params_grid["data_source"] = ["interTVA"]  # ["ABIDE", "interTVA"]
    params_grid["auto_encoded"] = [True]  # [True, False]
    params_grid[
        "fold"
    ] = 10  # More folds, means better representation, but more resources used

    for fold in range(1, params_grid["fold"] + 1):
        for data_source in params_grid["data_source"]:
            for delta in params_grid["delta"]:
                for soft_thresh in params_grid["soft_thresh"]:
                    for auto_encoded in params_grid["auto_encoded"]:
                        if data_source == "ABIDE":
                            run_slurm_job_regression(
                                data_source,
                                "gyrification",
                                delta,
                                soft_thresh,
                                fold,
                                script_name="regression.py"
                                if auto_encoded
                                else "regression_raw.py",
                            )
                        elif data_source == "interTVA":
                            for modality in params_grid["modality"]:
                                run_slurm_job_regression(
                                    data_source,
                                    modality,
                                    delta,
                                    soft_thresh,
                                    fold,
                                    script_name="regression.py"
                                    if auto_encoded
                                    else "regression_raw.py",
                                )

