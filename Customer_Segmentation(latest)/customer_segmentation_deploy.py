#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:54:34 2022

@author: farihahisa
"""

import pickle
import os
from tensorflow.keras.models import load_model

#%% Path

DATASET_TRAIN_PATH = os.path.join(os.getcwd(),'datasets','train.csv')
DATASET_TEST_PATH = os.path.join(os.getcwd(),'datasets','new_customers.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','model.h5') 
LOG_PATH = os.path.join(os.getcwd(),'log_customer')

#%% loading of settings or models

# onehot encoder

# encoder = pickle.load(f)
# encoded_docs = [encoder(d, vocab_size) for d in df.text]

# if you are using deep learning

model = load_model(MODEL_SAVE_PATH)
model.summary()

#%% deployment

model.predict(X_test)

outcome = model.predict(X_test)

print(np.argmax(outcome))
print(diabetes_chance[np.argmax(outcome)])

