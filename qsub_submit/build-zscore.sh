#$ -l tmem=12G
#$ -l h_vmem=30G
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/others/SuperMudi/results/qsub_job
cd /cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/main
source activate iqt
source /share/apps/source_files/cuda/cuda-10.1.source

python build_iso.py --rebuild -jn build_iso --no_subject 5 0 --approach IsoSRUnet -nf 16 -nk 2 -mt 2 -nl 3 -c 1 -mnp 100 -ip 8 8 8 -op 16 16 16 -opt 8 8 8 -es 4 4 4 -est 4 4 4 -od zscore

# (41, 46, 28) -> 10x11x7
