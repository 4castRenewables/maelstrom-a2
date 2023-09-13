
## Create venv on JUWELS

Create venv
```bash
module purge
module load Stages/2023
module load GCCcore/.11.3.0
module load Python/3.10.4
module load CUDA/11.7
mkdir /p/scratch/deepacf/maelstrom/maelstrom_data/ap2/venvs/ap2_deberta
python -m venv /p/scratch/deepacf/maelstrom/maelstrom_data/ap2/venvs/ap2_deberta
```
and activate
```
source /p/scratch/deepacf/maelstrom/maelstrom_data/ap2/venvs/ap2_deberta/bin/activate
```