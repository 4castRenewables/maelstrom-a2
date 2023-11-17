## Install apptainer

See [this git doc](https://github.com/apptainer/apptainer/blob/main/INSTALL.md) for installation of apptainer.

## How to create a apptainer image based on a docker image
Adopt `recipe.def` to your liking, then call:
```bash
sudo apptainer build NAME_OF_IMAGE.sif recipe.def
```

## How to use on JSC in Juelich
Copy apptainer image to the cluster and execute:
```bash
install-apptainer-jupyter-kernel.sh KERNEL_NAME apptainer_IMAGE USER_KERNEL_DIR_STEM
```

Private kernels need to be saved in folder: `/p/home/jusers/ehlert1/juwels/.local/share/jupyter/kernels/`.
For boot camp in folder: `/p/project/training2223/.local/share/jupyter/kernels/`.

Check for available kernels via `!jupyter kernelspec list`.

Check [this README](https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template/-/tree/singularity) or [this JSC documentation](https://docs.jupyter-jsc.fz-juelich.de/github/kreuzert/jupyter-jsc-notebooks/blob/documentation/03-HowTos/install-singularity-jupyter-kernel.ipynb) for more details.

Setup in `kernel.json` (`--nv` crucial for GPU support, see [apptainer documentation](https://docs.sylabs.io/guides/3.1/user-guide/cli/singularity_exec.html)):
```json
{
 "argv": [
   "apptainer",
   "exec",
   "--nv",
   "--cleanenv",
   "/p/project/training2223/venv_apps/venv_ap2/ap2/ap2.sif",
   "python3",
   "-m",
   "ipykernel",
   "-f",
   "{connection_file}"
 ],
 "language": "python",
 "display_name": "ap2"
}
```
Check access rights if sharing kernel with other users!
## Start session on cluster with image
```bash
srun -N1 -p develgpus --account deepacf --time=02:00:00 --gres gpu:1 --pty apptainer shell --nv APPTAINER_IMAGE.sif
```

## Start from Dockerfile (not working for me)
see https://github.com/apptainer/singularity/issues/1537
and https://groups.google.com/a/lbl.gov/g/singularity/c/1OzBUM6uEow/m/d8om_vFeAwAJ

```bash
sudo docker run -d -p 5000:5000 --restart=always --name registry registry:2
sudo docker push hugging_face/scratch
sudo docker tag hugging_face/scratch localhost:5000/hugging_face/scratch
```
And apptainer file contains:
```text
Bootstrap: docker
Registry: http://localhost:5000
Namespace:
From: hugging_face
```