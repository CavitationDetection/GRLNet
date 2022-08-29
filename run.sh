#!/bin/bash
###
 # @Description: 
 # @Author: Yu Sha
 # @Date: 2021-08-17 09:57:39
 # @LastEditors: Yu Sha
 # @LastEditTime: 2022-05-25 16:50:42
### 

#SBATCH --job-name=No_GFL
#SBATCH --error=err_no_GFL.log
#SBATCH --output=out_no_GFL.log
#SBATCH --reservation srivastava-shared
#SBATCH --nodes=1                                         # set the number of nodes
#SBATCH --partition=sleuths                               # set partition
#SBATCH --nodelist=geralt                                # set node (turbine,vane,speedboat,jetski,scuderi,tussock,geralt#SBATCH --gres=gpu:1  #SBATCH --reservation deepthinkers)  
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24    
#SBATCH --time=10000:00:00                                # run time of task


srun python3 train.py

