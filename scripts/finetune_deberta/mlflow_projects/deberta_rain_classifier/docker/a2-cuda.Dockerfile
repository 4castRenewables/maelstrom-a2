FROM --platform="linux/arm64/v8" nvcr.io/nvidia/pytorch:24.01-py3

# RUN python${PYTHON_VERSION} -m ensurepip --upgrade
# RUN sed -ie '$d' /opt/nvidia/deepstream/deepstream-6.3/entrypoint.sh
# RUN sed -ie '$a /opt/nvidia/nvidia_entrypoint.sh $@' /opt/nvidia/deepstream/deepstream-6.3/entrypoint.sh
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
# ENV GDAL_VERSION="3.4.1"
COPY scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/requirements_cuda.txt /opt/deberta_rain_classifier/
WORKDIR /opt/deberta_rain_classifier
RUN pip install --upgrade pip

RUN apt-get upgrade -y
RUN aarch64-linux-gnu-gcc --version
# RUN gcc-aarch64-linux-gnu --version
RUN apt-get upgrade gcc-aarch64-linux-gnu
ENV DISABLE_NUMCODECS_AVX2=1
ENV DISABLE_NUMCODECS_SSE2=1
RUN echo ${DISABLE_NUMCODECS_SSE2}
RUN pip index versions a2
RUN pip install -r requirements_cuda.txt
# RUN pip install --ignore-installed -r requirements_cuda.txt

RUN python${PYTHON_VERSION} -c 'import a2, torch, torchvision, ipykernel'
RUN python${PYTHON_VERSION} -c 'from torch._C._distributed_c10d import ProcessGroupNCCL'
WORKDIR /opt/pytorch

# Install poetry
