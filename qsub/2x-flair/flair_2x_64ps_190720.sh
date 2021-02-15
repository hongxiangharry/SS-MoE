#$ -l tmem=12G
#$ -pe gpu 2
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_2x.py -jn 190720_Aniso_flair_64ps_3nl --approach AnisoUnet -es 32 32 16 -est 32 32 16 -ip 64 64 32 -op 64 64 64 -opt 32 32 32 -nf 16 -nk 2 -mt 2 -nl 3 -c 1 -bs 1 # batchsize = 1
python main_flair_2x.py -jn 190720_Aniso_flair_64ps_6nl --approach AnisoUnet -es 32 32 16 -est 32 32 16 -ip 64 64 32 -op 64 64 64 -opt 32 32 32 -nf 16 -nk 2 -mt 2 -nl 6 -c 1 -bs 1 # batchsize = 1