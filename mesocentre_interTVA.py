import os
import os.path as op
import datetime
import feature_extraction_ABIDE as fea
import json

email = "mmahaut@ensc.fr"
logs_dir = "/scratch/mmahaut/scripts/logs"
python_path = "python3.7"
slurm_dir = "/scratch/mmahaut/scripts/slurm"
code_dir = "/scratch/mmahaut/scripts/INT_fMRI_processing"
subs_list_file_path = (
    "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list.json"
)
script_name = "feature_extraction_ABIDE.py"

subs_list_file = open(subs_list_file_path)
subject_list = json.load(subs_list_file)


for subject in subject_list:
    job_name = "{}_extraction".format(subject)
    slurmjob_path = op.join(slurm_dir, "{}.sh".format(job_name))
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
        # number of nodes for this job
        fh.writelines("#SBATCH --nodes=1\n")
        # number of cores for this job
        fh.writelines("#SBATCH --ntasks-per-node=1\n")  # ??
        # email alerts
        fh.writelines("#SBATCH --mail-type=BEGIN,END\n")
        fh.writelines("#SBATCH --mail-user={}\n".format(email))
        # making sure group is ok for data sharing within group
        batch_cmd = (
            "export SUBJECTS_DIR=/scratch/mmahaut/data/intertva/downloaded_preprocessed\n"
            "chmod +x /scratch/mmahaut/scripts/INT_fMRI_processing/for_redistribution_files_only/run_find_eig.sh\n"
            'eval "$(/scratch/mmahaut/tools/Anaconda3/bin/conda shell.bash hook)"\n'
            + "conda activate ABIDE\n"
            + "{} {}/{} {}".format(python_path, code_dir, script_name, subject)
        )

        fh.writelines(batch_cmd)

    os.system("sbatch %s" % slurmjob_path)
