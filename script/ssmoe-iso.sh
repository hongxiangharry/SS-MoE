#!/usr/bin/env bash

JOB_ID="default"
SSD_PATH="<TARGET_DIR_TO_UNZIP>/"${JOB_ID}"/patch" # could be an SSD
ZIP_FILE="<SOURCE_ZIP_FILE_DIR>/iso-(8, 8, 8)-(4, 4, 4)-4-None-0.zip"

patch_num=400000
job_name="ssmoe-iso"
cross_validation=0

# Build training data
cd <PATH/TO/PROJECT/DIRECTORY>/SS-MoE-miccai/main
source activate ssmoe

python build_iso.py -jn build_iso -cvf ${cross_validation} --no_subject 4 1 --approach IsoSRUnet -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

# Train and eval stage 1 (S1)
echo "Run SS-MoE on SuperMUDI datasets"
echo "Make the dir at "${SSD_PATH}
mkdir -p $SSD_PATH
echo "Unzipping data ..."
unzip -q "$ZIP_FILE" -d "$SSD_PATH"

python main_iso.py -jn ${job_name} -cvf ${cross_validation} --no_subject 4 1 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

# Train and test stage 2 & 3 (S2+S3)

python main_S2_train_label.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 5 0 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

python main_S2_train_fc.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 4 1 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

python main_S3_train_label.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 5 0 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

python main_S3_train_decoder.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 4 1 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

python main_S3_test.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 4 1 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

function finish {
    rm -rf ${SSD_PATH}
}

trap finish EXIT ERR