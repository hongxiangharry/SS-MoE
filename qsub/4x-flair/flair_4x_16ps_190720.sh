#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_4x.py -jn 190720_Aniso_flair_4x_16ps_3nl --approach AnisoUnet -es 8 8 2 -est 8 8 2 -ip 16 16 4 -op 16 16 16 -opt 8 8 8 -nf 16 -nk 2 -mt 2 -nl 3 -c 1
python main_flair_4x.py -jn 190720_Aniso_flair_4x_16ps_4nl --approach AnisoUnet -es 8 8 2 -est 8 8 2 -ip 16 16 4 -op 16 16 16 -opt 8 8 8 -nf 16 -nk 2 -mt 2 -nl 4 -c 1