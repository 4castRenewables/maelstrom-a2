# Testing two NLP neural networks for sentiment classification of Reddit comments

## Overview
Here,we classify Reddit comments as neutral or emotional. Original data taken from [datasets by hugging face](https://huggingface.co/datasets/go_emotions), which is categorized in 28 labels originally. We keep the **neutral** class and combine others to form the **emotional** class.

We then train a deep neural network based on 1D convolutional layers and pre-trained embeddings called [TextCNN](https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html) on the data. In addition,
we train the transformer based model [DeBERTa-small](https://huggingface.co/microsoft/deberta-v3-small) on the data. In addition, we use the same architecture just scaled up termed [deberta-base](https://huggingface.co/microsoft/deberta-v3-base).

For evaluation, we use the f1-score, which gives us the following results including a few runs with varying hyperparameters:

| model name | f1 score  <br> macro avg | f1 score <br> weighted avg |
| ------ | ------ | ------ |
| Text CNN | 0.69  | 0.75 |
| deberta-v3-small | 0.73  | 0.78 |
| -- learning rate/10 | 0.73  | 0.78 |
| -- weight decay *10 | 0.73  | 0.78 |
| deberta-v3-base | 0.74  | 0.79 |
| -- fine tune | -  | - |
| -- learning rate/10 | 0.73  | 0.78 |

'-' here means that model didn't learn to generalize.
## Setup
We trained our models on the jupyter notebook environment [jupyter-jsc](https://jupyter-jsc.fz-juelich.de/hub/home) ([intro slides](https://docs.jupyter-jsc.fz-juelich.de/github/FZJ-JSC/jupyter-jsc-notebooks/blob/master/Jupyter-JSC_supercomputing-in-the-browser.pdf)) on the Juelich supercomputer.
By setting up our own Jupyter CONDA kernel that is based on our personal conda environment on the cluster, new packages can be installed on your private conda environment on the cluster and become immediately available on jsc.

To reproduce our results one needs to:
1. Install your own version of conda on the cluster. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) provides
a light-weight solution.
2. Clone this repo on the cluster preferably in your home so you can easily access contained notebooks.
3. Prepare your cluster environment for install. Apparently, the cuda package should be visible when installing pytorch.
   ```bash
   module purge
   module load Stages/2020
   module load CUDA/11.3
   module load cuDNN/8.2.1.32-CUDA-11.3
   ```
4. Instantiate our environment on the cluster from files provided in folder *environment*. Two options are provided:
   1. ```bash
      conda create --name myenv --file environment/spec-file.txt
      ```
      see [the conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments) for more info.

      OR
   2. ```bash
      conda env create -f environment/environment.yml
      ```
      The first line of the yml file sets the new environment's name ([see docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)).
5. Create a jupyter kernel based on the environment that can be accessed by jsc as described in [this notebook](https://docs.jupyter-jsc.fz-juelich.de/github/FZJ-JSC/jupyter-jsc-notebooks/blob/master/001-Jupyter/Create_JupyterKernel_conda.ipynb).
6. On [jupyter-jsc](https://jupyter-jsc.fz-juelich.de/hub/home), **Add a new JupyterLab** with System=JUWELS and Partition=develgpus for access to the debug gpu queue, which has minimal waiting time. Computations can only take 2h, which is fine for this project.
7. Load the desired notebook and pick your user-created kernel.








