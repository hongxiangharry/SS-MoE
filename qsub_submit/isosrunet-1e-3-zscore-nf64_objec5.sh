#$ -l tmem=12G
#$ -l tscratch=20G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/others/SuperMudi/results/qsub_job

SSD_PATH="/scratch0/harrylin/"$JOB_ID"/patch"
ZIP_FILE="/cluster/project0/IQT_Nigeria/others/SuperMudi/patch/iso-(8, 8, 8)-(4, 4, 4)-5-zscore.zip"

echo "Run IQT on SuperMUDI datasets"
echo "Make the dir at "${SSD_PATH}
mkdir -p $SSD_PATH
echo "Unzipping data ..."
unzip -q "$ZIP_FILE" -d "$SSD_PATH"

cd /cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/main
source activate iqt
source /share/apps/source_files/cuda/cuda-10.1.source

python main_iso_test.py --retrain -jn iso_lr1e-3_p888_sbj5_z_nf64 --approach IsoSRUnet -nf 64 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 64 -lr 1e-3 -j $JOB_ID -vs 0 -ntp 270000 -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4 -od zscore

function finish {
    rm -rf /scratch0/harrylin/$JOB_ID
}

trap finish EXIT ERR