#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_2x.py -jn 190720_Aniso_flair_4ps_3nl_2048 --approach AnisoUnet -es 2 2 1 -est 2 2 1 -ip 4 4 2 -op 4 4 4 -opt 2 2 2 -nf 16 -nk 2 -mt 2 -nl 3 -c 1 -bs 2048
python main_flair_2x.py -jn 190720_Aniso_flair_4ps_2nl_2048 --approach AnisoUnet -es 2 2 1 -est 2 2 1 -ip 4 4 2 -op 4 4 4 -opt 2 2 2 -nf 16 -nk 2 -mt 2 -nl 2 -c 1 -bs 2048