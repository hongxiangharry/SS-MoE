#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_1_2_12.py -jn 180720_Aniso_flair_lre-3 --approach AnisoUnet -es 24 12 2 -nf 16 -nk 2 -mt 2 -c 1
python main_flair_1_2_12.py -jn 180720_SRUnet_flair_lre-3 --approach SRUnet -es 24 12 2 -nf 16 -nk 2 -mt 2 -c 1