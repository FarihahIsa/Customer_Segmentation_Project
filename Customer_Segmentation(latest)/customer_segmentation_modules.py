#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:14:20 2022

@author: farihahisa
"""

import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

#%% Path

DATASET_TRAIN_PATH = os.path.join(os.getcwd(),'datasets','train.csv')
DATASET_TEST_PATH = os.path.join(os.getcwd(),'datasets','new_customers.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','model.h5') 
LOG_PATH = os.path.join(os.getcwd(),'log_customer')

#%%

X_train = pd.read_csv(DATASET_TRAIN_PATH)
X_test = pd.read_csv(DATASET_TEST_PATH, header=None)

#%%

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def CleaningData(self, data):
        
        # Fill non-null values
        X_train['Ever_Married'].fillna(method='ffill', inplace=True)
        X_train['Graduated'].fillna(method='bfill', inplace=True)
        X_train['Profession'].fillna(method='ffill', inplace=True)
        X_train['Work_Experience'].fillna(method='bfill',inplace=True)
        X_train['Family_Size'].fillna(method='bfill', inplace=True )
        
        
        # Convert object in column into int
        x_train = X_train[['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession', 
                           'Work_Experience', 'Spending_Score', 'Family_Size']]

        x_train['Gender'] = x_train['Gender'].map({'Female':0, 'Male':1})
        x_train['Ever_Married'] = x_train['Ever_Married'].map({'No':0, 'Yes':1})
        x_train['Graduated'] = x_train['Graduated'].map({'No':0, 'Yes':1})
        x_train['Profession'] = x_train['Profession'].map({'Artist':0,'Doctor':1, 
                                                           'Engineer':2, 'Entertainment':3,
                                                           'Executive':4, 'Healthcare':5,
                                                           'Homemaker':6, 'Lawyer':7,
                                                           'Marketing':8})
        x_train['Spending_Score'] = x_train['Spending_Score'].map({'Average':0, 'Low':1, 
                                                                   'High':2})
        
    def FeatureSelection(self, data):
        
        X = x_train
        y = X_train['Segmentation']
        
    def DataPreprocessing (self, data):
        
        y = OneHotEncoder(sparse=False).fit_transform(np.expand_dims(y, axis=-1))
        
        # train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
class ModelCreation():
       
    def ModelNN(self, data):
        
        model = Sequential()
        model.add(Input(shape=(8)))
        model.add(Dense(80, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(4, activation='softmax'))
        
        return model
    
    def ModelCompile(self, data):
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        
        # early stopping callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)
        
        # callbacks
        log_files= os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

        tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
        
class ModelTraining():
    
    def Modelfitting(self, data):
        hist = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), 
                         callbacks=[tensorboard_callback, early_stopping_callback])

        hist.history.keys()
        training_loss = hist.history['loss']
        training_acc = hist.history['accuracy']
        validation_loss = hist.history['val_loss']
        validation_acc = hist.history['val_accuracy']
        
        # plot graph of accuracy and loss
        plt.figure()
        plt.plot(training_loss)
        plt.plot(validation_loss) #validation loss
        plt.title('training loss and validation loss')
        plt.xlabel('epoch')
        plt.ylabel('Cross entropy loss')
        plt.legend(['training loss', 'validation loss'])
        plt.show()

        plt.figure()
        plt.plot(training_acc)
        plt.plot(validation_acc) #validation acc
        plt.title('training acc and validation acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['training acc', 'validation acc'])
        plt.show()
        
class ClassificationReport():
    
    def ConfusionMatrix(self, data):
        pred_x = model.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(pred_x, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        
    def ClassificationReport(self, data):
        cr = classification_report(np.argmax(y_test, axis=1), np.argmax(pred_x, axis=1))
        print(cr)
    
#%% 

eda = ExploratoryDataAnalysis()
mc = ModelCreation()
mt = ModelTraining()
cr = ClassificationReport()