# INSTRUCTIONS

## Cluster setup

- Go inside the remote GPU server: (personal step, skip this if you are already in the GPU server)

  ```bash
  ssh borg-cluster
  ```

- Load the necessary modules:

  ```bash
  module use /usr/local/borgwardt/modulefiles
  module load python/3.7.7
  module load cuda/10.1
  ```

- Setup the GPU:

  - Check the available GPUs:

    ```bash
    nvidia-smi
    ```

  - Set `CUDA_VISIBLE_DEVICES` to a free GPU. For instance:

    ```bash
    export CUDA_VISIBLE_DEVICES=0
    ```

## Running the code

- Download and go inside the repository:

  ```bash
  git clone https://github.com/kaanoktay/SeFT_Transformer.git
  cd SeFT_Transformer
  ```

- Run the main file as below:

  ```bash
  poetry install
  poetry run main --batch_size 16 --num_epochs 10
  ```
