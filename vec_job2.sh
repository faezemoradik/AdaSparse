
## Run below line in the terminal to submit the job
## SEED can be set to different values, such as 0,1,2


sbatch --export=ALL,SEED=0,ME="AdaSparse",KAPPA=1e1 vec_job1.sh
sbatch --export=ALL,SEED=0,ME="LCAdaSparse",KAPPA=1e1 vec_job1.sh
sbatch --export=ALL,SEED=0,ME="topk",KAPPA=1.8e3 vec_job1.sh
sbatch --export=ALL,SEED=0,ME="UniSparse",KAPPA=16.0 vec_job1.sh
sbatch --export=ALL,SEED=0,ME="varreduced",KAPPA=1e-3 vec_job1.sh
