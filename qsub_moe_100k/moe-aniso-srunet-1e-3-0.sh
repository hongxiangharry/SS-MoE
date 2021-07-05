#$ -l tmem=12G
#$ -l tscratch=20G
#$ -l gpu=true
#$ -pe gpu 2
#$ -l hostname='!scooter'
#$ -l hostname='!pepe'
#$ -l hostname='!uncledeadly'
#$ -l hostname='!webb'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/others/SuperMudi/results/qsub_job

SSD_PATH="/scratch0/yukuzhou/"$JOB_ID"/patch"
ZIP_FILE="/cluster/project0/IQT_Nigeria/others/SuperMudi/patch/aniso-(16, 16, 8)-(8, 8, 4)-4-None-0.zip"

echo "Run IQT on SuperMUDI datasets"
echo "Make the dir at "${SSD_PATH}
mkdir -p $SSD_PATH
echo "Unzipping data ..."
unzip -q "$ZIP_FILE" -d "$SSD_PATH"

hostname

cd /cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_moe_iso_2/main
source activate iqt
source /share/apps/source_files/cuda/cuda-10.1.source

python main_moe_aniso.py -jn moe_aniso_nf64_100k_0 --no_subject 4 1 --approach MoEAnisoUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp 100000 -mnp 100 -ip 16 16 8 -op 16 16 16 -opt 8 8 8 -es 8 8 4 -est 8 8 4

function finish {
    rm -rf /scratch0/harrylin/$JOB_ID
}

trap finish EXIT ERR