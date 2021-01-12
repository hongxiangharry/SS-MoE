#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_6x.py -jn 120720_Aniso_flair6x_lre-3 --no_subject 15 15 --approach AnisoUnet -nf 16 -nk 2 -mt 2 -c 1
python main_flair_6x.py -jn 120720_SRUnet_flair6x_lre-3 --no_subject 15 15 --approach SRUnet -nf 16 -nk 2 -mt 2 -c 1