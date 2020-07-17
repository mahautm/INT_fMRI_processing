import os

for i in range(100):
    cmd = "freeview -f $SUBJECTS_DIR/fsaverage5/surf/rh.white:overlay=/media/sf_StageINT/data/gyrcheck/USM_0050516_segmented_fsaverage5_{0}.gii --viewport 3d -ss /media/sf_StageINT/data/gyr/{0}".format(
        i
    )
    os.system(cmd)
