# Train and evaluate on cluster with JUBE

## Local installation requirements
Install apptainer via [repo instructions](https://github.com/apptainer/apptainer/blob/main/INSTALL.md).



## Environment for H100
### Locations images
- Tensorflow containers
    - MI250x: container - rocm/tensorflow:rocm5.7-tf2.13-dev
    - H100: container - nvcr.io/nvidia/tensorflow:23.10-tf2-py3
- PyTorch containers
    - MI250x: container - rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1
    - H100: container - nvcr.io/nvidia/pytorch:23.10-py3
### Pull and build containers (on cluster)
```bash
container=nvcr.io/nvidia/pytorch:23.10-py3
export APPTAINER_CACHEDIR=$PROJECT/maelstrom/$USER/apptainercache/
apptainer pull --tmpdir $PROJECT/maelstrom/$USER/apptainertmpdir/ docker://${container}
```
### Install packages locally and pull into container
```bash
apptainer run --nv pytorch_23.10-py3.sif
pip install -I --prefix=$(pwd)/h100_packages/ -r ../../scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/requirements.txt
```
### Test setup locally 
```bash
apptainer run --nv pytorch_23.10-py3.sif
export PYTHONPATH=/p/project/deepacf/maelstrom/ehlert1/a2/cluster/benchmarks3.7/h100_packages/local/lib/python3.10/dist-packages:$PYTHONPATH
python3 -c "import transformers"
```

## Environment for MI250
### Locations images
- Tensorflow containers
    - MI250x: container - rocm/tensorflow:rocm5.7-tf2.13-dev
    - H100: container - nvcr.io/nvidia/tensorflow:23.10-tf2-py3
- PyTorch containers
    - MI250x: container - rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1
    - H100: container - nvcr.io/nvidia/pytorch:23.10-py3
### Pull and build containers (on cluster)
```bash
container=rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1
export APPTAINER_CACHEDIR=$PROJECT/maelstrom/$USER/apptainercache/
apptainer pull --tmpdir $PROJECT/maelstrom/$USER/apptainertmpdir/ docker://${container}
```
### Install packages locally and pull into container
```bash
apptainer run --nv pytorch_rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1.sif
python3 -m pip install -I --prefix=$(pwd)/mi250_packages/ -r../../scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/requirements_rocm.txt
```
## Environment for Grace Hopper (E4)
Grace hopper gpu is mounted in an ARM machine. To build a docker image that can be run on arm using non-arm architecture, the following docker image should be executed
```bash
docker run --rm -t arm64v8/ubuntu:latest uuname -m
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes # This step will execute the registering scripts
docker run --rm -t arm64v8/ubuntu uname -m # Testing the emulation environment
```
Our base image is built on a nvidia provided image, it can locally be run by calling
```
# run docker in detached mode
docker run -t -d --ipc=host -e HOME=$HOME -e USER=$USER nvcr.io/nvidia/pytorch:24.01-py3
```

Now the docker image can be build, set IMAGE_TYPE=ap2armcuda in Makefile and run 
```
make build-docker
```
and upload the image to docker hub
```
make upload-to-dockerhub
```
Now the image should be available on the Grace Hopper machine and run via `/home/kristian/Projects/a2/scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/run_docker_e4.sh`. See jube script for more details.


## Setup environment
3. 
# TODO: clarify difference ".nc" and ".csv" file.
```bash
sacctmgr show user femmerich withassoc
nvcc --version
eval "$(/home/kehlert/grace-hopper/bin/conda shell.bash hook)"
source activate /home/kehlert/.conda/envs/a2-gracehopper
```

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

## Known issues:
### Local apptainer installation
```bash
INFO:    Starting build...
FATAL:   While performing build: conveyor failed to get: while converting reference: loading image from docker engine: Error response from daemon: client version 1.22 is too old. Minimum supported API version is 1.24, please upgrade your client to a newer version
make: *** [Makefile:193: build-apptainer] Error 255
```
You can trick client by manually overriding minimum version, see [issue](https://github.com/containers/skopeo/issues/2202#issuecomment-1908830671).