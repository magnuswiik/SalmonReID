#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=ie-iel
#SBATCH --time=01:00:00
#SBATCH --nodes=1              # 1 compute nodes
#SBATCH --ntasks-per-node=1    # 1 mpi process each node
#SBATCH --mem=16000            # 16GB - in megabytes
#SBATCH --job-name="detector_train"
#SBATCH --output=/cluster/home/magnuwii/masterthesis/IDUNfiles/fasterrrcnn/output/detection_train.out
#SBATCH --mail-user=magnuwii@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load Anaconda3/2022.10
conda activate fisketrening
python main.py
