# Sentiment-Analysis-with-Bidirectional-LSTM-Model

## A - Sentiment Analysis Project trained on the 50000 IMDB Movie Reviews Dataset

This project aims to classify inputs into positive or negative classes. This dataset can be found at
https://ai.stanford.edu/~amaas/data/sentiment/

### How to Use

1. Download the data from the link above.
2. The dataset comes in 2 folders, `train` and `test`, each with two subfolders, named `pos` and `neg`. Place `train` and `test` into the main folder and run main.py to obtain 2 csv files, `testreviews.csv` and `trainreviews.csv`.
3. Open `LSTM Sentiment Classifier.ipynb` and run it to train the model. Ideally, the computer should be connected to a GPU.

## B - Sentiment Analysis Project trained on 3.6 million Amazon Reviews Dataset
This project uses a similar architecture to A, except that it is performed on the 3.6 million Amazon Reviews Dataset. Trained on a Nvidia RTX 3070 GPU for more than a week, the classifier obtains a test score of about 94%.
