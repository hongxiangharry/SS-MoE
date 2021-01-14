#$ -l tmem=12G
#$ -l h_vmem=30G
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/others/SuperMudi/results/qsub_job
cd /cluster/project0/IQT_Nigeria/others/SuperMudi/code/IQT_supermudi/main
source activate iqt

python build_iso.py -jn build_iso --approach IsoSRUnet -nf 16 -nk 2 -mt 2 -nl 3 -c 1 -ip 4 4 4 -op 8 8 8 -opt 4 4 4 -es 2 2 2 -est 2 2 2
