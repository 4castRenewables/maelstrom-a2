ARG PYTHON_VERSION=3.10

# NOTE: When updating ROCM version, make sure to change it
# for both layers as well ass the `poetry add source` command.
FROM rocm/dev-ubuntu-22.04:5.6-complete as builder

ARG ROCM_VERSION
ARG PYTHON_VERSION
RUN apt-get update \
 && apt-get install -y software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      # Installs gcc, required by HDBSCAN
      build-essential \
      # Required by cartopy
      libgeos3.10.2 \
      libgeos-dev \
      # Install opencv via apt to get required libraries
      python3-opencv \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-venv
RUN apt --installed list
RUN python3.10 --version
# RUN python${PYTHON_VERSION} -m ensurepip --upgrade
RUN pip${PYTHON_VERSION} --help

# Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python${PYTHON_VERSION} -
ENV PATH=/opt/poetry/bin:${PATH}
ENV POETRY_VIRTUALENVS_CREATE=false

COPY README.md/ /opt/a2/
COPY pyproject.toml /opt/a2/
COPY poetry.lock /opt/a2/
COPY src/a2/ /opt/a2/src/a2
COPY scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/requirements_rocm.txt /opt/deberta_rain_classifier/

WORKDIR /opt/deberta_rain_classifier

ENV VIRTUAL_ENV=/opt/venv
RUN python${PYTHON_VERSION} -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
RUN pip install -r requirements_rocm.txt
#  && poetry remove torch --group=torch-cpu 
#  \
#  && poetry source add --priority=supplemental pytorch-rocm https://download.pytorch.org/whl/rocm5.6 \
#  && poetry add \
#     -vvv \
#     --source pytorch-rocm \
#     torch==$(poetry show torch | awk '/version/ { print $3 }') \
#     torchvision==$(poetry show torchvision | awk '/version/ { print $3 }') \
#  && poetry install --only=main,notebooks

# Delete Python cache files
# WORKDIR /venv
# RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

# FROM rocm/dev-ubuntu-22.04:5.6-complete

# ARG PYTHON_VERSION

# ARG PATH=/venv/bin:${PATH}
# ENV PATH=/venv/bin:${PATH}

# ENV GIT_PYTHON_REFRESH=quiet

# # Copy venv and repo
# COPY --from=builder /venv /venv
# COPY --from=builder /opt/a2 /opt/a2

# RUN apt-get update \
#  && DEBIAN_FRONTEND=noninteractive \
#     apt-get install -y --no-install-recommends  \
#       # Installs gcc, required by HDBSCAN
#       build-essential \
#       # Required by cartopy
#       libgeos3.10.2 \
#       libgeos-dev \
#       # Install opencv via apt to get required libraries
#       python3-opencv \
#       python${PYTHON_VERSION} \
#       python${PYTHON_VERSION}-dev \
#  && apt-get clean \
#  && rm -rf /var/lib/apt/lists/*

# # Inlcude ROCm SMI libraries for python
ENV PYTHONPATH=/opt/rocm/libexec/rocm_smi/:$PYTHONPATH

# # Check if all packages successfully installed by importing
# # RUN which python
RUN python${PYTHON_VERSION} --version
RUN pip list
RUN python${PYTHON_VERSION} -c 'import rsmiBindings'
RUN python${PYTHON_VERSION} -c 'import pynvml'
RUN python${PYTHON_VERSION} -c 'import a2, torch, torchvision, ipykernel'
RUN python${PYTHON_VERSION} -c 'import torch.distributed.distributed_c10d as c10d; assert c10d._NCCL_AVAILABLE, "NCCL not available"'
RUN python${PYTHON_VERSION} -c 'from torch._C._distributed_c10d import ProcessGroupNCCL'

ENTRYPOINT ["python${PYTHON_VERSION}"]
