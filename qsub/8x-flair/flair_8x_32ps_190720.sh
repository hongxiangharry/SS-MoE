#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_8x.py -jn 190720_Aniso_flair_8x_32ps_4nl --approach AnisoUnet -es 16 16 2 -est 16 16 2 -ip 32 32 4 -op 32 32 32 -opt 16 16 16 -nf 16 -nk 2 -mt 2 -nl 4 -c 1
python main_flair_8x.py -jn 190720_Aniso_flair_8x_32ps_5nl --approach AnisoUnet -es 16 16 2 -est 16 16 2 -ip 32 32 4 -op 32 32 32 -opt 16 16 16 -nf 16 -nk 2 -mt 2 -nl 5 -c 1