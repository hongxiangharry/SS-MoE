#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_8x.py -jn 190720_Aniso_flair_8x_8ps_4nl_256 --approach AnisoUnet -es 4 4 1 -est 4 4 1 -ip 8 8 1 -op 8 8 8 -opt 4 4 8 -nf 16 -nk 2 -mt 2 -nl 4 -c 1 -bs 256