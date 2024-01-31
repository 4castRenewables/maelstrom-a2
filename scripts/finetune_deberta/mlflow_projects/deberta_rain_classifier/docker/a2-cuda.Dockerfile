FROM nvcr.io/nvidia/pytorch:24.01-py3

# RUN python${PYTHON_VERSION} -m ensurepip --upgrade
# RUN sed -ie '$d' /opt/nvidia/deepstream/deepstream-6.3/entrypoint.sh
# RUN sed -ie '$a /opt/nvidia/nvidia_entrypoint.sh $@' /opt/nvidia/deepstream/deepstream-6.3/entrypoint.sh
RUN sed --help
RUN /opt/nvidia/nvidia_entrypoint.sh && python --version
RUN pip --version

COPY scripts/finetune_deberta/mlflow_projects/deberta_rain_classifier/requirements_cuda.txt /opt/deberta_rain_classifier/
WORKDIR /opt/deberta_rain_classifier

RUN pip install --upgrade pip
RUN pip install -r requirements_cuda.txt

RUN python${PYTHON_VERSION} -c 'import a2, torch, torchvision, ipykernel'
RUN python${PYTHON_VERSION} -c 'from torch._C._distributed_c10d import ProcessGroupNCCL'
WORKDIR /opt/pytorch

# Install poetry
