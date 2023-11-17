#!/bin/bash
# see singularity_images/README_singularity_image.md for details
# Author: Katharina Höflich
# Repository: https://github.com/FZJ-JSC/jupyter-jsc-notebooks

KERNEL_NAME=${1}
SINGULARITY_IMAGE=${2}

[[ ! -z "$KERNEL_NAME" ]] || echo "Provide a Jupyter kernel name, please."
[[ ! -z "$SINGULARITY_IMAGE" ]] || echo "Provide a Apptainer container image, please."

USER_KERNEL_DIR=${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}
mkdir -p ${USER_KERNEL_DIR} || exit

echo '{
 "argv": [
   "apptainer",
   "exec",
   "--nv",
   "--cleanenv",
   "'"${SINGULARITY_IMAGE}"'",
   "python",
   "-m",
   "ipykernel",
   "-f",
   "{connection_file}"
 ],
 "language": "python",
 "display_name": "'"${KERNEL_NAME}"'"
}' > ${USER_KERNEL_DIR}/kernel.json || exit
