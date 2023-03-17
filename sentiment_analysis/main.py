import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import keras
import random
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import os
code = "utf-8"

def read_file(directory):
    file = open(directory, "r", encoding=code)
    review = file.read()
    file.close()
    cleaned = re.sub("<.*?\/>", "", review)
    
    cleaned = re.sub("http:[^\s]*\s", "", cleaned)
    cleaned = re.sub("[^a-zA-Z.!?\s]+", "", cleaned)
    cleaned = re.sub(r"([.!?]+)", r" \1 ", cleaned)
    
    return cleaned

currentwd = os.path.dirname(os.path.realpath(__file__))

for i in ["test", "train"]:

    pos = f"{currentwd}/{i}/pos"
    neg = f"{currentwd}/{i}/neg"


    positivefiles = list(map(lambda x: f"{pos}/{x}", os.listdir(pos)))
    negativefiles = list(map(lambda x: f"{neg}/{x}", os.listdir(neg)))

    reviewspos = list(map(read_file, positivefiles))
    reviewsneg = list(map(read_file, negativefiles))

    df = pd.DataFrame(data= {"reviews":reviewspos + reviewsneg, "Positivity": [1 for i in range(len(positivefiles))] + [0 for i in range(len(negativefiles))]})
    df = df.sample(frac=1).reset_index(drop=True)
    print(df)
    df.to_csv(f"{currentwd}/{i}reviews.csv", encoding=code, index=False)