import os
import os.path as op
import datetime
import feature_extraction_ABIDE as fea
import json

email = "mmahaut@ensc.fr"
logs_dir = ""
python_path = ""
code_dir = ""
subs_list_file_path = "./url_preparation/subs_list.json"

subs_list_file = open(subs_list_file_path)
subject_list = json.load(subs_list_file)


for subject in subject_list:
    job_name = "{}_extraction".format(subject)
    slurmjob_path = op.join(logs_dir, "{}.sh".format(job_name))
    create_slurmjob_cmd = "touch {}".format(slurmjob_path)
    os.system(create_slurmjob_cmd)

    # processing_graph_dir = op.join(
    #     root_dir,
    #     "graph-based_analysis",
    #     "{}.{}.{}p_cf{:.01f}".format(hemisphere, roi, n_parcels, cf_w),
    # )
    # inputs_path = op.join(processing_graph_dir, "inputs.jl")

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
        fh.writelines("newgrp b125\n")
        # command to be executed
        script_name = "localizer_parallel_gsvc.py"
        batch_cmd = "{} {}/{} {}".format(python_path, code_dir, script_name, subject)
        fh.writelines(batch_cmd)

    os.system("sbatch %s" % slurmjob_path)
