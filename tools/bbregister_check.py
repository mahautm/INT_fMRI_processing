import os
import json

subject_list = "/media/sf_StageINT/INT_fMRI_processing/url_preparation/subs_list_asd.json"

subs_list_file = open(subject_list)
subs_list = json.load(subs_list_file)

for sub in subs_list:
    if not os.path.exists(
        "/media/sf_StageINT/data/regcheck/axial_{0}_nii.png".format(sub)
    ):
        cmd = "rsync mmahaut@login.mesocentre.univ-amu.fr:/scratch/mmahaut/data/abide/intermediary/{0}/{0}_func_preproc.nii.gz /media/sf_StageINT/data/datareg/{0}/{0}_func_preproc.nii.gz ".format(
            sub
        )
        cmd += " && rsync mmahaut@login.mesocentre.univ-amu.fr:/scratch/mmahaut/data/abide/downloaded_preprocessed/{0}/surf/lh.white /media/sf_StageINT/data/datareg/{0}/lh.white".format(
            sub
        )
        # cmd += " && rsync mmahaut@login.mesocentre.univ-amu.fr:/scratch/mmahaut/data/abide/downloaded_preprocessed/{0}/mri/T1.mgz /media/sf_StageINT/data/datareg/{0}/T1.mgz".format(
        #     sub
        # )
        if not os.path.exists("/media/sf_StageINT/data/datareg/{0}/".format(sub)):
            os.makedirs("/media/sf_StageINT/data/datareg/{0}/".format(sub))
        os.system(cmd)
        print("{} rsync done".format(sub))
        cmd = "freeview -f /media/sf_StageINT/data/datareg/{0}/lh.white -v /media/sf_StageINT/data/datareg/{0}/{0}_func_preproc.nii.gz -viewport axial -ss /media/sf_StageINT/data/regcheck/axial_{0}_nii".format(
            sub
        )
        os.system(cmd)

        # cmd = "freeview -f /media/sf_StageINT/data/datareg/{0}/lh.white -v /media/sf_StageINT/data/datareg/{0}/T1.mgz -viewport axial -ss /media/sf_StageINT/data/regcheck/axial_{0}_T1".format(
        #     sub
        # )
        # os.system(cmd)
    else:
        print("images already exist for " + sub)
