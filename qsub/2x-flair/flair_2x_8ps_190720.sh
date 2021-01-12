#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_2x.py -jn 190720_Aniso_flair_8ps_3nl_256 --approach AnisoUnet -es 4 4 2 -est 4 4 2 -ip 8 8 4 -op 8 8 8 -opt 4 4 4 -nf 16 -nk 2 -mt 2 -nl 3 -c 1 -bs 256
#python main_flair_2x.py -jn 190720_Aniso_flair_16ps_4nl --approach AnisoUnet -es 8 8 4 -est 8 8 4 -ip 16 16 8 -op 16 16 16 -opt 8 8 8 -nf 16 -nk 2 -mt 2 -nl 4 -c 1