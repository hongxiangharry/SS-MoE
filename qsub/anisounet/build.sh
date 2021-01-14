#$ -l tmem=12G
#$ -l h_vmem=30G
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/others/SuperMudi/results/qsub_job
cd /cluster/project0/IQT_Nigeria/others/SuperMudi/code/IQT_supermudi/main
source activate iqt

python build.py -jn build_aniso --approach AnisoUnet -nf 16 -nk 2 -mt 2 -nl 3 -c 1 -ne 100 -bs 16
