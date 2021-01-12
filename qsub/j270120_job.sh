#$ -l tmem=12G
#$ -l gpu=true
#$ -l hostname='!gonzo-605-2'
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/IQT_Nigeria/research/baseline_t1t2/qsub_job
cd ~/IQT_baseline
source activate usiqt
python main_multimodal_k4.py -jn 250220_Aniso_2modals_lre-3 --no_subject 15 15 --approach AnisoUnet -es 16 16 4 -nf 16 -nk 2 -mt 2 -c 5
