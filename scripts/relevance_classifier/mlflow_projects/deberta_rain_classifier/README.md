
## Create venv on JUWELS

Create venv
```bash
module purge
module load Stages/2023
module load GCCcore/.11.3.0
module load Python/3.10.4
cd /p/scratch/deepacf/maelstrom/maelstrom_data/ap2/venvs
python -m venv <venv-name>
```
and activate
```
source <venv-name>/bin/activate
```