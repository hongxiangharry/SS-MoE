#$ -l tmem=12G
#$ -l tscratch=20G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/others/SuperMudi/results/qsub_job

SSD_PATH="/scratch0/harrylin/"$JOB_ID"/patch"
ZIP_FILE="/cluster/project0/IQT_Nigeria/others/SuperMudi/patch/aniso-(16, 16, 8)-(8, 8, 4)-4-zscore.zip"

echo "Run IQT on SuperMUDI datasets"
echo "Make the dir at "${SSD_PATH}
mkdir -p $SSD_PATH
echo "Unzipping data ..."
unzip -q "$ZIP_FILE" -d "$SSD_PATH"

cd /cluster/project0/IQT_Nigeria/others/SuperMudi/code/IQT_supermudi/main
source activate iqt

python main_eval.py --retrain -jn aniso_lr1e-3_p16168_sbj4_z_nf64 --no_subject 4 1 --approach AnisoUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -ntp 270000 -mnp 100 -ip 16 16 8 -op 16 16 16 -opt 8 8 8 -es 8 8 4 -est 8 8 4 -od zscore

function finish {
    rm -rf /scratch0/harrylin/$JOB_ID
}

trap finish EXIT ERR