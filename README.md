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

## Running A2 on Mantik
Currently, the following MLflow projects exist to run A2 models with Mantik
- Split data and train model of baseline DeBERTa classifier, see `scripts/relevance_classifier/mlflow_projects/deberta_rain_classifier/.
### Setup on Juwels Cluster
Here, we execute our code in a venv environment. For setup of a venv with the necessary packages installed, can be found in README.md of respective MLflow project folder.
### Setup on Mantik platform GUI
In your project, create an `Experiment` to log data to and `Code` that refers to this repository.
### Create Run
In the `Run` form adopt parameter values of the `MLflow` project file and compute backend file to your setup.

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

### Skip tests that require large datasets

Use `--skip_optional` when using pytest to skip these tests.


### Testing images
Tests are based on the package [pytest-mpl](https://github.com/matplotlib/pytest-mpl).
Running tests is simplified through pre-defined commands via the Makefile. To run all tests, simply call

```shell
make test
```
Baseline images are generated via
```shell
make test-generate-images
```

Tests including image comparisons are run when including the option
```shell
make test-view-images
```

Note, that `torch` is not installed by default, use `poetry install --with torch-cpu` to install it on your local machine (without gpu). If not installed,
 torch will be set to `None`, which may lead to unexepected errors.
## Poetry

### Known issues
- [Invalid hashes](https://stackoverflow.com/questions/71001968/python-poetry-install-failure-invalid-hashes):
    Search for problematic file `find ~/.cache/pypoetry -name numpy-1.22.2-cp38-cp38-macosx_10_14_x86_64.whl` and remove it.


## Ideas

### Relevance classification as additional step

```bash
. test_build_dataset_relevance_classifier.sh
. test_finetune_deberta_classifier_relevance.sh
. test_predict_deberta_classifier_relevance.sh
. test_build_dataset_rain_classifier.sh
. test_finetune_deberta_classifier_rain.sh
. test_predict_deberta_classifier_rain.sh
```

### Paint weather map
Embed Tweets as RGB (or similar) on a grid (unstructured?!/average embeddings?!/take most informative Tweet?!) and treat as image to predict rain map.

### First paper idea
* Introduce keyword distribution -> histogram plot
* Motivate certainty in prediction with softmax output
* Results: confusion matrix and/or AUC plot

### Noisy labels
* Precipitation forecasts are notoriously difficult. Therefore models are built to systematically overpredict rain. For this project, comparing ERA5 data with data from nearby weather stations showed a precision of predicting rain of only 35%.

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

### Relevance Classifier -> Rain classifier
* Relevance classifier:
    * Used to classify tweets as "relevant" for identification of "raining"/"not raining" (could a human deduct this from the Tweet)
    * Dataset:
        * Labeling1:
            * "relevant":
                * Tweets matching keyword
                * ~250k Tweets (stratified by "raining")
            * "irrelevant"
                * Tweets are sample of Tweets only with location (no keyword matching applied).
                * 2020-02-13T - 2020-02-15T
                * 250k Tweets
        * Labeling2:
            * Use LLM (ChatGpt4/Falcon) to build dataset
            * Use prompt to let LLM classify tweets;
                * Raining likelihood: float -> not raining (0) - raining (1)
                * How certain assessment (sufficient precipitation-related content present): float -> no information (0) - perfectly clear information (1)
            * Use [relevance ai](https://relevanceai.com/bulk-run) to pass csv of Tweets to be classified by engineerd prompt
            * ~100s examples could be sufficient
        * Splits:
            * Dataset: 500k Tweets
            * Train 60%, Validate 20%, Test 20%
    * Model1:
        * Finetune DeBertA-v3-base classifier
    * Model2:
        * Finetune LLM (Falcon), which should reduce information retrieval time (?!)
            * Backup model for uncertain Tweets?
* Rain classifier:
    * Classify Tweets as "raining" / "not raining" when above `rain_threshold` (e.g., 0.1 mm).
    * Dataset:
        * Include:
            * Tweets classified as "relevant" by `relevance classifier`
        * Exclude:
            * Tweets used to train the `relevance classifier`
            * Tweets near (e.g., 1km) weather station -> build seperate test set
        * Splits:
            * Dataset: ?? Tweets
            * Train 60%, Validate 20%, Test 20%
