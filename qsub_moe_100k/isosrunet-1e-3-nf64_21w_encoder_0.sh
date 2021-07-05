#$ -l tmem=12G
#$ -l tscratch=20G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/others/SuperMudi/results/qsub_job

SSD_PATH="/scratch0/yukun/"$JOB_ID"/patch"
ZIP_FILE="/cluster/project0/IQT_Nigeria/others/SuperMudi/patch/iso-(8, 8, 8)-(4, 4, 4)-4-None-0.zip"

patch_num=400000
job_name=iso_lr1e-3_p888_sbj4_nf64_decoder_4_cv0_20210621
cross_validation=0
echo "Run IQT on SuperMUDI datasets"
echo "Make the dir at "${SSD_PATH}
mkdir -p $SSD_PATH
echo "Unzipping data ..."
unzip -q "$ZIP_FILE" -d "$SSD_PATH"

cd /cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/main
source activate iqt
source /share/apps/source_files/cuda/cuda-10.1.source

python main_S2_train_label.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 5 0 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

python main_S2_train_fc.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 4 1 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

python main_S3_train_label.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 5 0 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

python main_S3_train_decoder.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 4 1 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

python main_S3_test.py --retrain -jn ${job_name} -cvf ${cross_validation} --no_subject 4 1 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp ${patch_num} -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4

function finish {
    rm -rf /scratch0/yukun/$JOB_ID
}

trap finish EXIT ERR