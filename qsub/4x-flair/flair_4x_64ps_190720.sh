#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_4x.py -jn 190720_Aniso_flair_4x_64ps_3nl_1 --approach AnisoUnet -es 32 32 8 -est 32 32 8 -ip 64 64 16 -op 64 64 64 -opt 32 32 32 -nf 16 -nk 2 -mt 2 -nl 3 -c 1 -bs 1 # batchsize = 4
python main_flair_4x.py -jn 190720_Aniso_flair_4x_64ps_6nl_1 --approach AnisoUnet -es 32 32 8 -est 32 32 8 -ip 64 64 16 -op 64 64 64 -opt 32 32 32 -nf 16 -nk 2 -mt 2 -nl 6 -c 1 -bs 1 # batchsize = 4