#$ -l tmem=12G
#$ -l tscratch=20G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/others/SuperMudi/results/qsub_job

SSD_PATH="/scratch0/harrylin/"$JOB_ID"/patch"
ZIP_FILE="/cluster/project0/IQT_Nigeria/others/SuperMudi/patch/iso-(16, 16, 16)-(8, 8, 8).zip"

echo "Run IQT on SuperMUDI datasets"
echo "Make the dir at "${SSD_PATH}
mkdir -p $SSD_PATH
echo "Unzipping data ..."
unzip -q "$ZIP_FILE" -d "$SSD_PATH"

cd /cluster/project0/IQT_Nigeria/others/SuperMudi/code/IQT_supermudi/main
source activate iqt

python main_iso.py --retrain -jn isosrunet_1e-4_es8 --approach IsoSRUnet -nf 16 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 16 -lr 1e-4 -j $JOB_ID -es 8 8 8

function finish {
    rm -rf /scratch0/harrylin/$JOB_ID
}

trap finish EXIT ERR