# Train and evaluate on cluster with JUBE

## Setup environment
1. Install Miniconda (see [docs](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)).
2. Create environment `a2-benchmarking` and install required packages.
    * For Nvidia GPUs:
        ```bash
        conda create -n a2-benchmarking python=3.10
        conda activate a2-benchmarking
        . install.sh
        ```
    * For AMD GPUs:
        ```bash
        conda create -n a2-amd python=3.10
        conda activate a2-amd
        . install_amd.sh
        ```
3. Create `env.sh` file, which exports all global variables required by Mantik (see [Mantik](https://cloud.mantik.ai/) docs for more details).

*Note*, Nvidia setup is based on Cuda Version 11.7, which is the most up to date version present on JUWELS/E4.
*Note2*, benchmarks were run with python package a2 version 0.3.
## Submit training
1. Make sure module JUBE (jube) is loaded

    * On JUWELS:
        ```bash
        module load JUBE PyYAML
        ```
    * On E4:
        ```bash
        module use /opt/share/users/testusr/modulefiles
        module load jube
        ```
2. Submit training jobs using `jube`
    * On JUWELS
        ```bash
        jube run jube_training.yaml --tag jwc medium
        ```
    * On E4 (Nvidia)
        ```bash
        jube run jube_training.yaml --tag e4 medium
        ```
    * On E4 (AMD)
        ```bash
        jube run jube_training.yaml --tag e4 e4amd medium
        ```
    * Available tags:
        * `jwc`: JUWELS Cluster
        * `jwb`: JUWELS Booster
        * `e4`: E4 queues
        * `e4amd`: E4 AMD queue (`e4` also required)
        * `test`: Used for debugging
        * `medium`: Medium sized training/evaluation set
        * `large`: Large training/evaluation set
        * `hps`: Hyper parameter tuning setup
        * `benchs`: Benchmarking setup (Starts multiple runs with same parameters at once)

3. Display results
    Print benchmarking results for runs (via run `id`):
    ```bash
    jube result -a -u jube_training.yaml ap2_run --id 43 46
    ```
## Submit evaluation
1. Submit evaluation jobs using `jube`
    * On JUWELS
        ```bash
        jube run jube_evaluation.yaml --tag jwc medium
        ```
    * On E4 (Nvidia)
        ```bash
        jube run jube_evaluation.yaml --tag e4 medium
        ```
    * On E4 (AMD)
        ```bash
        jube run jube_evaluation.yaml --tag e4 e4amd medium
        ```
    * Available tags:
        * `jwc`: JUWELS Cluster
        * `jwb`: JUWELS Booster
        * `e4`: E4 queues
        * `e4amd`: E4 AMD queue (`e4` also required)
        * `medium`: Medium sized training/evaluation set
        * `large`: Large training/evaluation set

2. Display results
    Print benchmarking results for runs (via run `id`):
    ```bash
    jube result -a -u jube_evaluation.yaml ap2_eval --id 11 12
    ```

## Training without jube
Alternatively, job script `submit_train.sh` can be used to train the model on the cluster (via `run_benchmark.sh`).