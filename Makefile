.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

ROOT_DIR = $(PWD)
NOTEBOOKS_DIR = $(ROOT_DIR)/notebooks
JSC_DIR = $(NOTEBOOKS_DIR)/containers/jsc
CERTAIN_TRANSFORMER_DIR = $(ROOT_DIR)/modelling/certain-transformer-rain-prediction
CERTAIN_TRANSFORMER_JSC_DIR = certain_transformer

JSC_USER = ${MANTIK_UNICORE_USERNAME}
JSC_PROJECT = ${MANTIK_UNICORE_PROJECT}
JSC_SSH = $(JSC_USER)@juwels-cluster.fz-juelich.de
JSC_SSH_PRIVATE_KEY_FILE = -i $(HOME)/.ssh/jsc

IMAGE_TYPE = llama

ifeq ($(IMAGE_TYPE), llama)
	POETRY_GROUPS := llama-chatbot torch
	IMAGE_NAME := ap2python3p10llama
else
	POETRY_GROUPS := TRAIN
	IMAGE_NAME := ap2python3p10
endif

install:
	poetry install

build-python:
	# Remove old build
	# rm -rf dist/
	# poetry build -f wheel
	rm requirements.txt
	poetry export --with "${POETRY_GROUPS}" > requirements.txt

build-conda-env: build-python
	conda create -n $(IMAGE_NAME) python=3.10
	$(CONDA_ACTIVATE) "$(IMAGE_NAME)"
	pip install -r requirements.txt
	pip install --user ipykernel
	python -m ipykernel install --user --name=$(IMAGE_NAME)

build-docker: build-python
	sudo docker build -t $(IMAGE_NAME):latest -f mlflow/Dockerfile .

build-apptainer: build-python
	sudo apptainer build --force mlflow/$(IMAGE_NAME).sif mlflow/recipe.def

build: build-docker build-apptainer

clean-poetry:
	rm -rf .venv
	rm poetry.lock

upload:
	rsync -Pvra $(JSC_SSH_PRIVATE_KEY_FILE) \
		mlflow/$(IMAGE_NAME).sif \
		$(JSC_SSH):/p/project/$(JSC_PROJECT)/$(JSC_USER)/$(IMAGE_NAME).sif

deploy: build upload

build-jsc-kernel: build-python
	sudo apptainer build --force \
		$(JSC_DIR)/jupyter-kernel.sif \
		$(JSC_DIR)/jupyter_kernel_recipe.def

build-certain-transformer-image: build-python
	sudo apptainer build --force \
		$(CERTAIN_TRANSFORMER_DIR)/jupyter-kernel.sif \
		$(CERTAIN_TRANSFORMER_DIR)/jupyter_kernel_recipe.def

upload-certain-transformer-image:
	rsync -Pvra \
		$(CERTAIN_TRANSFORMER_DIR)/jupyter-kernel.sif \
		$(JSC_SSH):/p/scratch/$(JSC_PROJECT)/maelstrom/maelstrom_data/ap2/singularity_images/$(CERTAIN_TRANSFORMER_JSC_DIR)/	.sif

define JSC_KERNEL_JSON
{
 "argv": [
   "singularity",
   "exec",
   "--nv",
   "--cleanenv",
   "/p/scratch/$(JSC_PROJECT)/maelstrom/maelstrom_data/ap2/singularity_images/jupyter-kernel.sif",
   "python3",
   "-m",
   "ipykernel",
   "-f",
   "{connection_file}"
 ],
 "language": "python",
 "display_name": "$(IMAGE_NAME)"
}
endef

export JSC_KERNEL_JSON

upload-jsc-kernel:
	# Copy kernel image file
	rsync -Pvra \
		$(JSC_DIR)/jupyter-kernel.sif \
		$(JSC_SSH):/p/scratch/$(JSC_PROJECT)/maelstrom/maelstrom_data/ap2/singularity_images/jupyter-kernel.sif

	# Create kernel.json file
	$(eval KERNEL_FILE := $(JSC_DIR)/kernel.json)
	echo "$${JSC_KERNEL_JSON}" > $(KERNEL_FILE)

	# Upload kernel.json file
	$(eval KERNEL_PATH="/p/home/jusers/$(JSC_USER)/juwels/.local/share/jupyter/kernels/$(IMAGE_NAME)/")
	ssh $(JSC_SSH) "mkdir -p $(KERNEL_PATH)"
	rsync -Pvra  $(KERNEL_FILE) $(JSC_SSH):$(KERNEL_PATH)
	rm $(KERNEL_FILE)

deploy-jsc-kernel: build-jsc-kernel upload-jsc-kernel
