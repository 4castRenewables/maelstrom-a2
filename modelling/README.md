# Collection of models used for this project

## Overview
The following models are available.
### reddit-sentiment-classification
Here,we classify proxy data in the form of Reddit comments as neutral or emotional. We train a deep neural network based on 1D convolutional layers and pre-trained embeddings called [TextCNN](https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html) on the data. In addition,
we train the transformer based model [DeBERTa](https://huggingface.co/microsoft/deberta-v3-small) on the dta.
