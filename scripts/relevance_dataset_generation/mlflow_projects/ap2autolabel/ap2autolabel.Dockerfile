FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG PYTHON_VERSION=3.10
RUN apt-get update \
 && apt-get install -y software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
 apt-get install -y --no-install-recommends \
     libgdal-dev \
     build-essential \
     python${PYTHON_VERSION}-dev
RUN /opt/nvidia/nvidia_entrypoint.sh && python --version
RUN pip --version
RUN pip install --upgrade pip

RUN apt-get upgrade -y
RUN pip install a2[benchmarks,xarray-extra,deberta,notebooks] accelerate 
RUN pip install refuel-autolabel transformers bitsandbytes

RUN python${PYTHON_VERSION} -c 'import a2, torch, torchvision, ipykernel'
RUN python${PYTHON_VERSION} -c 'from autolabel import LabelingAgent, AutolabelDataset, get_data'
WORKDIR /opt/pytorch

# Install poetry
