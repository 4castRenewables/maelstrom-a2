.ONESHELL:
ACTIVE_ENV = source ~/.bashrc
SHELL := /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
ROOT_DIR = $(PWD)
NOTEBOOKS_DIR = $(ROOT_DIR)/notebooks
JSC_DIR = $(NOTEBOOKS_DIR)/containers/jsc
CERTAIN_TRANSFORMER_DIR = $(ROOT_DIR)/modelling/certain-transformer-rain-prediction
CERTAIN_TRANSFORMER_JSC_DIR = certain_transformer
MANTIK_UNICORE_USERNAME = ehlert1
MANTIK_UNICORE_PROJECT = deepacf

JSC_USER = ${MANTIK_UNICORE_USERNAME}
JSC_PROJECT = ${MANTIK_UNICORE_PROJECT}
JSC_SSH = $(JSC_USER)@juwels22.fz-juelich.de#juwels-cluster.fz-juelich.de
JSC_SSH_PRIVATE_KEY_FILE = -i $(HOME)/.ssh/jsc

IMAGE_TYPE = bootcamp2023
KERNEL_IMAGE_DEFINITION_FILENAME := jupyter_kernel_recipe
POETRY_GROUPS := ""
POETRY_EXTRAS := ""
ifeq ($(IMAGE_TYPE), llama)
	POETRY_EXTRAS := llama-chatbot torch
	IMAGE_NAME := ap2python3p10llama
else ifeq ($(IMAGE_TYPE), llamachat)
	POETRY_EXTRAS := ""
	IMAGE_NAME := llama-chat
	KERNEL_IMAGE_DEFINITION_FILENAME := llama-chat
	KERNEL_PATH := /p/project/training2330/ehlert1/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/training2330/ehlert1/jupyter/images/
	KERNEL_DISPLAY_NAME := ap2_llama
else ifeq ($(IMAGE_TYPE), HFfinetuningBnB)
	POETRY_EXTRAS := ""
	IMAGE_NAME := HFfinetuningBnB
	KERNEL_IMAGE_DEFINITION_FILENAME := HFfinetuningBnB
	KERNEL_PATH := /p/project/training2330/ehlert1/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/training2330/ehlert1/jupyter/images/
	KERNEL_DISPLAY_NAME := ap2_HF-LLM-BnB
else ifeq ($(IMAGE_TYPE), bootcamp2023)
	JSC_DIR := bootcamp2023/solutions/
	POETRY_GROUPS := train
	IMAGE_NAME := bootcamp2023
	KERNEL_IMAGE_DEFINITION_FILENAME := $(IMAGE_NAME)
	KERNEL_PATH := /p/project/training2330/ehlert1/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/training2330/ehlert1/jupyter/images/
	KERNEL_DISPLAY_NAME := ap2
else
	POETRY_EXTRAS := TRAIN
	IMAGE_NAME := ap2python3p10
endif

ifndef KERNEL_PATH
KERNEL_PATH := "/p/home/jusers/$(JSC_USER)/juwels/.local/share/jupyter/kernels/$(IMAGE_NAME)/"
endif

ifndef JSC_IMAGE_FOLDER
JSC_IMAGE_FOLDER := "/p/scratch/$(JSC_PROJECT)/maelstrom/maelstrom_data/ap2/singularity_images/"
endif

ifndef KERNEL_DISPLAY_NAME
KERNEL_DISPLAY_NAME := "$(IMAGE_NAME)"
endif

install:
	poetry install

build-python:
	# Remove old build
	# rm -rf dist/
	# poetry build -f wheel
	rm requirements.txt
	poetry export -f requirements.txt --without-hashes --output requirements.txt --extras "${POETRY_EXTRAS}" --with "${POETRY_GROUPS}"

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

publish-package:
	poetry publish --build

publish-patch:
	poetry version patch
	$(MAKE) publish-package

clean-poetry:
	rm -rf .venv
	rm poetry.lock

test:
	poetry run pytest --mpl --mpl-hash-library=/home/kristian/Projects/a2/src/tests/data/mpl_baselines/hash_json.json --mpl-baseline-path=/home/kristian/Projects/a2/src/tests/data/mpl_baselines/ --mpl-generate-summary=html --cov=a2 --record-mode=once --cov-report=term-missing -n 14 src/tests/

test-view-images:
	poetry run pytest --mpl --mpl-hash-library=/home/kristian/Projects/a2/src/tests/data/mpl_baselines/hash_json.json --mpl-baseline-path=/home/kristian/Projects/a2/src/tests/data/mpl_baselines/ --cov=a2 --cov-report=term-missing --record-mode=once --mpl-generate-summary=html src/tests/

test-generate-images:
	poetry run pytest --mpl-generate-hash-library=src/tests/data/mpl_baselines/hash_json.json --mpl-generate-path=src/tests/data/mpl_baselines/ --record-mode=once src/tests/

upload:
	rsync -Pvra $(JSC_SSH_PRIVATE_KEY_FILE) \
		mlflow/$(IMAGE_NAME).sif \
		$(JSC_SSH):/p/project/$(JSC_PROJECT)/$(JSC_USER)/$(IMAGE_NAME).sif

deploy: build upload

build-jsc-kernel:
	sudo apptainer build --force \
		$(JSC_DIR)/$(IMAGE_NAME).sif \
		$(JSC_DIR)/$(KERNEL_IMAGE_DEFINITION_FILENAME).def

build-certain-transformer-image: build-python
	sudo apptainer build --force \
		$(CERTAIN_TRANSFORMER_DIR)/$(IMAGE_NAME).sif \
		$(CERTAIN_TRANSFORMER_DIR)/$(KERNEL_IMAGE_DEFINITION_FILENAME).def

upload-certain-transformer-image:
	rsync -Pvra \
		$(CERTAIN_TRANSFORMER_DIR)/$(IMAGE_NAME).sif \
		$(JSC_SSH):$(JSC_IMAGE_FOLDER)/$(CERTAIN_TRANSFORMER_JSC_DIR)/$(IMAGE_NAME).sif

define JSC_KERNEL_JSON
{
 "argv": [
   "singularity",
   "exec",
   "--nv",
   "--cleanenv",
   "$(JSC_IMAGE_FOLDER)/$(IMAGE_NAME).sif",
   "python3",
   "-m",
   "ipykernel",
   "-f",
   "{connection_file}"
 ],
 "language": "python",
 "display_name": "$(KERNEL_DISPLAY_NAME)"
}
endef

export JSC_KERNEL_JSON

sync-bootcamp-data:
	. ~/.bashrc && rsynctojuwels /home/kristian/Projects/a2/data/bootcamp2023/ /p/project/training2330/a2/data/bootcamp2023/

upload-jsc-kernel:
	echo $(KERNEL_PATH)
	# Copy kernel image file
	rsync -Pvra \
		$(JSC_DIR)/$(IMAGE_NAME).sif \
		$(JSC_SSH):$(JSC_IMAGE_FOLDER)/$(IMAGE_NAME).sif

	# Create kernel.json file
	$(eval KERNEL_FILE := $(JSC_DIR)/kernel.json)
	echo "$${JSC_KERNEL_JSON}" > $(KERNEL_FILE)

	# Upload kernel.json file
	ssh $(JSC_SSH) "mkdir -p $(KERNEL_PATH)"
	rsync -Pvra  $(KERNEL_FILE) $(JSC_SSH):$(KERNEL_PATH)
	rm $(KERNEL_FILE)

deploy-jsc-kernel: build-jsc-kernel upload-jsc-kernel

