# A2 - Social Media Data Analysis

## Organisation of repository
Code base and notebooks are generally split into dataset preparation, analysis and training.

### Datasets
The project is most fundamentally based on Tweets. To download Tweets see `notebooks/dataset/tweet_download.ipynb` while preprocessing of the Tweets is done in `notebooks/dataset/preprocess_dataset.ipynb`.

Regarding precipitation data, we have looked at
* the era5 dataset ([see copernicus](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview)). See notebook
* UK weather stations ([from here](https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-rain-obs/dataset-version-202207))
* radar data ([from the Met office](https://catalogue.ceda.ac.uk/uuid/82adec1f896af6169112d09cc1174499))

## Apptainer images
Training on HPC is mostly done with jupyter notebooks with jupyter kernels based on apptainer images.

### Install apptainer
See [this git doc](https://github.com/apptainer/apptainer/blob/main/INSTALL.md) for installation of apptainer.

### Building and deploying the Jupyter kernel
1. Prerequisites:
Create a private SSH file ~/.ssh/jsc (~/.ssh/e4) and upload its public counterpart
to JuDoor
2. Build Singularity image with package and ipykernel installed
```shell
make build-jsc-kernel
```
3. Upload the image and the kernel.json file:
```shell
make upload-jsc-kernel
```

Note, make sure that you are using `python3` or `python` in the apptainer recipe and the `JSC_KERNEL_JSON` depending on your image environment!
### Running on Juwels (Booster)
1. Start a Jupyter lab via Jupyter JSC on the respective login node (i.e. Juwels or Juwels Booster).
2. Select the kernel (see above).

## Debugging
To show processing bottelenecks line_profiler is used. For memory profiling, we use the respective [memory_profiler](https://github.com/pythonprofilers/memory_profiler) package.

## Memory Debugging
The package `memory_profiler` provides a very handy decorator `@memory_profiler.profile()`, which should be added to functions that need to be profiled.

A baseline memory report of the decorated function can be generated via
```bash
python -m memory_profiler example.py
```

To see the memory consumption as a function of time, a plot can be generated with `memory_profiler`
```bash
mprof run --python python <script>
```
The results can then be visualized via
```bash
mprof plot
```

## CPU Debugging
The package `line_profiler` breaks down the workload per line and gives a concise overview of performance bottlenecks when analyzing a function.

The profiling can directly be done in jupyter-notebooks (see [this tutorial](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html)) by adding the line
```python
%load_ext line_profiler
```
and then executing
```python
%lprun -f FUNCTION FUNCTION(ARGUMENTS)
```
e.g.,
```python
%lprun -f add_station_precipitation add_station_precipitation(ds.sel(index=slice(1000)), df)
```
## Testing

### Testing images
Tests are based on the package [pytest-mpl](https://github.com/matplotlib/pytest-mpl).

Baseline images are generated via
```shell
poetry run pytest --mpl-generate-path=${MPL_BASELINE_PATH_A2}
```

Tests including image comparisons are run when including the option
```shell
poetry run pytest --mpl --mpl-baseline-path=${MPL_BASELINE_PATH_A2}
```


## Ideas

### Paint weather map
Embed Tweets as RGB (or similar) on a grid (unstructured?!/average embeddings?!/take most informative Tweet?!) and treat as image to predict rain map.

## Poetry

### Known issues
- [Invalid hashes](https://stackoverflow.com/questions/71001968/python-poetry-install-failure-invalid-hashes):
    Search for problematic file `find ~/.cache/pypoetry -name numpy-1.22.2-cp38-cp38-macosx_10_14_x86_64.whl` and remove it.
### First paper idea
* Introduce keyword distribution -> histogram plot
* Motivate certainty in prediction with softmax output
* Results: confusion matrix and/or AUC plot

### Clustering algorithms
* k-nearest neighbours (vote by k-nearest neigbour on member class/ average value for regressor)
### Automative active learning
* Add third label -> "information not provided" (INP)
* Maximize number of predicted labels "raining", "not raining" while retaining high AUC (by pushing unclear Tweets into INP)


* Include uncertainty in prediction of model and use clustering to refine this measure
    * Possibly embeddings already contain information of prediction probability

* Use precipitation as prelimenary labels
* Train classifier (e.g., using DeBERTa)
* Get uncertrainty in results
    * E.g., Multiple training runs -> take average of prediction "probability"
* Assign Tweet to cluster
    * Cluster results (e.g., embeddings of Tweets)
        * Optionally, use multiple runs of the clustering algorithm and/or multiple trained embeddings
    * Use density based cut, smarter version to detect clusters, ...
* Compute average uncertainty

* Use sentence embedding of the tweet, using universal sentence encoder or any more recent/fancy language model. I guess there are many models pre-trained already with Tweets.
* Apply a PCA analysis on embeddings to extract only the relevant components, and then with those, you can train a simple binary classifier that can give you a relevance score.
