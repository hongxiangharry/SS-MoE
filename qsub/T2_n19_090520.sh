#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline
source activate iqt

python main_t2_nigeria19_k8.py -jn 090520_Aniso8x_t2_lre-3 --no_subject 0 12 --approach AnisoUnet -es 16 16 2 -nf 16 -nk 2 -mt 2 -c 5
