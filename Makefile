.ONESHELL:
ACTIVE_ENV = source ~/.bashrc
SHELL := /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
ROOT_DIR = $(PWD)
NOTEBOOKS_DIR = $(ROOT_DIR)/notebooks
APPTAINER_DIR = $(NOTEBOOKS_DIR)/containers/jsc
CERTAIN_TRANSFORMER_DIR = $(ROOT_DIR)/modelling/certain-transformer-rain-prediction
CERTAIN_TRANSFORMER_JSC_DIR = certain_transformer
MANTIK_UNICORE_USERNAME = ehlert1
MANTIK_UNICORE_PROJECT = deepacf

JSC_USER = ${MANTIK_UNICORE_USERNAME}
E4_USER = kehlert
JSC_PROJECT = ${MANTIK_UNICORE_PROJECT}
JSC_SSH = $(JSC_USER)@juwels22.fz-juelich.de#juwels-cluster.fz-juelich.de
E4_SSH = $(E4_USER)@172.18.19.216
JSC_SSH_PRIVATE_KEY_FILE = -i $(HOME)/.ssh/jsc
E4_IMAGE_FOLDER := /home/kehlert/images/
IMAGE_TYPE = ap2armcuda
KERNEL_IMAGE_DEFINITION_FILENAME := jupyter_kernel_recipe
POETRY_GROUPS := ""
POETRY_EXTRAS := ""
APPTAINER_DIR := ""
DOCKER_BUILD_ARGS := "--progress=plain"
PATH_TWEETS = /home/kristian/Projects/a2/data/bootcamp2023/tweets/
tweets_prefix = tweets_2017_era5_normed_filtered
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
else ifeq ($(IMAGE_TYPE), ap2autolabel)
	POETRY_EXTRAS := ""
	APPTAINER_DIR := scripts/relevance_dataset_generation/mlflow_projects/ap2autolabel/
	DOCKER_DIR := $(APPTAINER_DIR)
	IMAGE_NAME := $(IMAGE_TYPE)
	KERNEL_IMAGE_DEFINITION_FILENAME := $(IMAGE_TYPE)
	KERNEL_PATH := /p/home/jusers/ehlert1/juwels/.local/share/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/deepacf/maelstrom/ehlert1/apptainer_images/
	KERNEL_DISPLAY_NAME := $(IMAGE_TYPE)
else ifeq ($(IMAGE_TYPE), HFfinetuningBnB)
	POETRY_EXTRAS := ""
	IMAGE_NAME := HFfinetuningBnB
	KERNEL_IMAGE_DEFINITION_FILENAME := HFfinetuningBnB
	KERNEL_PATH := /p/project/training2330/ehlert1/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/training2330/ehlert1/jupyter/images/
	KERNEL_DISPLAY_NAME := ap2_HF-LLM-BnB
else ifeq ($(IMAGE_TYPE), ap2falcon)
	APPTAINER_DIR := scripts/relevance_dataset_generation/mlflow_projects/
	POETRY_EXTRAS := ""
	IMAGE_NAME := $(IMAGE_TYPE)
	KERNEL_IMAGE_DEFINITION_FILENAME := $(IMAGE_TYPE)
	KERNEL_PATH := /p/home/jusers/ehlert1/juwels/.local/share/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/deepacf/maelstrom/ehlert1/apptainer_images/
	KERNEL_DISPLAY_NAME := $(IMAGE_TYPE)
else ifeq ($(IMAGE_TYPE), ap2rocm)
	APPTAINER_DIR := scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/apptainer/
	DOCKER_DIR := scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/docker/
	POETRY_EXTRAS := ""
	IMAGE_NAME := a2-rocm
	KERNEL_IMAGE_DEFINITION_FILENAME := $(IMAGE_TYPE)
	KERNEL_PATH := /p/home/jusers/ehlert1/juwels/.local/share/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/deepacf/maelstrom/ehlert1/apptainer_images/
	KERNEL_DISPLAY_NAME := $(IMAGE_TYPE)
else ifeq ($(IMAGE_TYPE), ap2cuda)
	APPTAINER_DIR := scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/apptainer/
	DOCKER_DIR := scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/docker/
	POETRY_EXTRAS := ""
	IMAGE_NAME := a2-cuda
	KERNEL_IMAGE_DEFINITION_FILENAME := $(IMAGE_TYPE)
	KERNEL_PATH := /p/home/jusers/ehlert1/juwels/.local/share/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/deepacf/maelstrom/ehlert1/apptainer_images/
	KERNEL_DISPLAY_NAME := $(IMAGE_TYPE)
else ifeq ($(IMAGE_TYPE), ap2armcuda)
	APPTAINER_DIR := scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/apptainer/
	DOCKER_DIR := scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/docker/
	POETRY_EXTRAS := ""
	IMAGE_NAME := a2-arm-cuda
	KERNEL_IMAGE_DEFINITION_FILENAME := $(IMAGE_TYPE)
	KERNEL_PATH := /p/home/jusers/ehlert1/juwels/.local/share/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/deepacf/maelstrom/ehlert1/apptainer_images/
	KERNEL_DISPLAY_NAME := $(IMAGE_TYPE)
else ifeq ($(IMAGE_TYPE), bootcamp2023)
	APPTAINER_DIR := bootcamp2023/solutions/
	POETRY_GROUPS := train
	IMAGE_NAME := bootcamp2023
	KERNEL_IMAGE_DEFINITION_FILENAME := $(IMAGE_NAME)
	KERNEL_PATH := /p/project/training2330/ehlert1/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/training2330/ehlert1/jupyter/images/
	KERNEL_DISPLAY_NAME := ap2
else ifeq ($(IMAGE_TYPE), ap2deberta)
	APPTAINER_DIR := scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/
	IMAGE_NAME := $(IMAGE_TYPE)
	KERNEL_IMAGE_DEFINITION_FILENAME := $(IMAGE_NAME)
	KERNEL_PATH := /p/home/jusers/ehlert1/juwels/.local/share/jupyter/kernels/$(IMAGE_NAME)/
	JSC_IMAGE_FOLDER := /p/project/deepacf/maelstrom/ehlert1/apptainer_images/
	KERNEL_DISPLAY_NAME := ap2deberta
else
	POETRY_EXTRAS := TRAIN
	IMAGE_NAME := ap2python3p10
endif

ifndef KERNEL_PATH
KERNEL_PATH := "/p/home/jusers/$(JSC_USER)/juwels/.local/share/jupyter/kernels/$(IMAGE_NAME)/"
endif

ifndef JSC_IMAGE_FOLDER
JSC_IMAGE_FOLDER := "/p/scratch/$(JSC_PROJECT)/maelstrom/maelstrom_data/ap2/apptainer_images/"
endif

ifndef KERNEL_DISPLAY_NAME
KERNEL_DISPLAY_NAME := "$(IMAGE_NAME)"
endif

install-default:
	poetry install --extras "deberta benchmarks xarray-extra" --sync --with torch-cpu

install-cuda-arm:
	poetry install --extras "benchmarks deberta notebooks" --sync --with torch-cpu 

test-apptainer-image-autolabel:
	apptainer run $(APPTAINER_DIR)/$(IMAGE_NAME).sif \
	python3 -c "from autolabel import LabelingAgent, AutolabelDataset, get_data"

test-apptainer-image-training:
	apptainer run $(APPTAINER_DIR)/$(IMAGE_NAME).sif \
	python3 scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/finetune_deberta_classifier.py \
    --filename_dataset_train /tmp/dataset_rain_classifier/dataset_split_thresh6M3//$(tweets_prefix)_train.csv \
    --filename_dataset_validate /tmp/dataset_rain_classifier/dataset_split_thresh6M3//$(tweets_prefix)_validate.csv \
    --filename_dataset_test /tmp/dataset_rain_classifier/dataset_split_thresh6M3//$(tweets_prefix)_test.csv \
    --model_path models/deberta-v3-small/ \
    --output_dir /tmp/trained_model/ \
    --trainer_name deep500 \
	--dataset_backend pandas \
	--ignore_tracking \
    --debug

test-apptainer-image-training-rocm:
	apptainer run scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/apptainer/a2-rocm.sif \
	python3 scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/finetune_deberta_classifier.py \
    --filename_dataset_train /tmp/dataset_rain_classifier/dataset_split_thresh6M3//$(tweets_prefix)_train.nc \
    --filename_dataset_validate /tmp/dataset_rain_classifier/dataset_split_thresh6M3//$(tweets_prefix)_validate.nc \
    --filename_dataset_test /tmp/dataset_rain_classifier/dataset_split_thresh6M3//$(tweets_prefix)_test.nc \
    --model_path models/deberta-v3-small/ \
    --output_dir /tmp/trained_model/ \
    --trainer_name deep500 \
    --debug

test-apptainer-image-training-cuda:
	apptainer run scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/apptainer/ap2deberta.sif \
	python3 scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/finetune_deberta_classifier.py \
    --filename_dataset_train /tmp/dataset_rain_classifier/dataset_split_thresh6M3//$(tweets_prefix)_train.nc \
    --filename_dataset_validate /tmp/dataset_rain_classifier/dataset_split_thresh6M3//$(tweets_prefix)_validate.nc \
    --filename_dataset_test /tmp/dataset_rain_classifier/dataset_split_thresh6M3//$(tweets_prefix)_test.nc \
    --model_path models/deberta-v3-small/ \
    --output_dir /tmp/trained_model/ \
    --trainer_name deep500 \
    --debug \
	--dataset_backend pandas

test-apptainer-image-split:
	apptainer run $(APPTAINER_DIR)/$(IMAGE_NAME).sif \
	python3 scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/build_dataset_rain_classifier.py \
    --filename_tweets $(PATH_TWEETS)/$(tweets_prefix).csv \
	--dataset_backend pandas \
    --key_precipitation_station tp_mm_station \
	--ignore_tracking \
    --output_dir /tmp/dataset_rain_classifier/

test-apptainer-image-split-rocm:
	apptainer run scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/apptainer/a2-rocm.sif \
	python3 scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/build_dataset_rain_classifier.py \
    --filename_tweets $(PATH_TWEETS)/$(tweets_prefix).nc \
    --key_precipitation_station tp_mm_station \
	--ignore_tracking \
    --output_dir /tmp/dataset_rain_classifier/

test-apptainer-image-run:
	apptainer run $(APPTAINER_DIR)/$(IMAGE_NAME).sif 

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

build-docker:
ifdef DOCKER_DIR	
	sudo docker buildx build $(DOCKER_BUILD_ARGS) -t $(IMAGE_NAME):latest -f $(DOCKER_DIR)/$(IMAGE_NAME).Dockerfile .
endif

build-apptainer:
	sudo apptainer build --force $(APPTAINER_DIR)/$(IMAGE_NAME).sif $(APPTAINER_DIR)/$(IMAGE_NAME).def 

upload-to-dockerhub:
	docker tag a2-arm-cuda:latest kristian4cast/ml:a2-cuda
	docker push kristian4cast/ml:a2-cuda

build: build-docker build-apptainer

publish-package:
	poetry publish --build

publish-patch:
	poetry version patch
	$(MAKE) publish-package

publish-minor:
	poetry version minor
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

upload-image:
	# Copy kernel image file
	rsync -Pvra \
		$(APPTAINER_DIR)/$(IMAGE_NAME).sif \
		$(JSC_SSH):$(JSC_IMAGE_FOLDER)/$(IMAGE_NAME).sif

upload-image-e4:
	# Copy kernel image file
	rsync -Pvra \
		$(APPTAINER_DIR)/$(IMAGE_NAME).def \
		$(E4_SSH):$(E4_IMAGE_FOLDER)/$(IMAGE_NAME).def

build-image:
	sudo apptainer build --force \
		$(APPTAINER_DIR)/$(IMAGE_NAME).sif \
		$(APPTAINER_DIR)/$(KERNEL_IMAGE_DEFINITION_FILENAME).def

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

upload-jsc-kernel: build-docker build-apptainer upload-image 
	echo $(KERNEL_PATH)
	rsync -Pvra \
		$(APPTAINER_DIR)/$(IMAGE_NAME).sif \
		$(JSC_SSH):$(JSC_IMAGE_FOLDER)/$(IMAGE_NAME).sif

	# Create kernel.json file
	$(eval KERNEL_FILE := $(APPTAINER_DIR)/kernel.json)
	echo "$${JSC_KERNEL_JSON}" > $(KERNEL_FILE)

	# Upload kernel.json file
	ssh $(JSC_SSH) "mkdir -p $(KERNEL_PATH)"
	rsync -Pvra  $(KERNEL_FILE) $(JSC_SSH):$(KERNEL_PATH)
	rm $(KERNEL_FILE)

deploy-jsc-kernel: build-image upload-jsc-kernel

deploy-image: build-image upload-image

deploy-image-e4: build-image upload-image-e4