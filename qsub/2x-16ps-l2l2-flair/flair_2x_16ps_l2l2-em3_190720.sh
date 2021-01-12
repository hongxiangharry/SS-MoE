#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline/main
source activate iqt

python main_flair_2x.py -jn 190720_Aniso_flair_16ps_3nl_l2l2-em3 --approach AnisoUnet -es 8 8 4 -est 8 8 4 -ip 16 16 8 -op 16 16 16 -opt 8 8 8 -nf 16 -nk 2 -mt 2 -nl 3 -c 1 -l l2l2 -lp 1e-3