#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:19:43 2021

@author: elham
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import deepchem as dc
from rdkit import Chem
import matplotlib.pyplot as plt
from deepchem.splits.splitters import ScaffoldSplitter,SpecifiedSplitter,RandomSplitter
from deepchem.models import GraphConvModel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
from keras.callbacks import ModelCheckpoint,EarlyStopping
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk
from deepchem.utils.evaluate import Evaluator
import os

####Read data from csv files , train, test1 and test2

input_data_train ="train_s.csv"
input_data_test= '../final_data/test1.csv'
input_data_test2= '../final_data/test2.csv'




####Deepchem feature extraxtion and data loaded
tasks=['reaction_energy']
featurizer = dc.feat.ConvMolFeaturizer()
loader = dc.data.CSVLoader(tasks=tasks, feature_field="reactant_smiles",featurizer=featurizer)
dataset_train=loader.featurize(input_data_train)
dataset_test1=loader.featurize(input_data_test)
dataset_test2=loader.featurize(input_data_test2)

##Normalized datasets
transformers_train = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_train, move_mean=True)
transformers_test1 = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_test1, move_mean=True)
transformers_test2 = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_test2,move_mean=True)


# Run before every test for reproducibility
def seed_all():
    np.random.seed(123)
    tf.random.set_seed(123)


####Split train data set to train and valid
seed_all()
splitter = dc.splits.RandomSplitter() #There is a "split" field in the dataset file where I  defined the training/valid/test set

train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset_train,frac_train = 0.8,frac_valid = 0.2, frac_test = 0.0,seed = 0)

####Transorm them 
train_dataset = transformers_train.transform(train_dataset)
valid_dataset= transformers_train.transform(valid_dataset)
test1_dataset = transformers_test1.transform(dataset_test1)
test2_dataset = transformers_test2.transform(dataset_test2)


#######

model_dir = './tf_chp_hp2'
logD_model = dc.models.GraphConvModel(n_tasks=1,batch_size=32, mode='regression', dropout=0.0,dense_layer_size=256,learning_rate=0.005,model_dir= model_dir,random_seed=0)
logD_model.restore('./tf_chp_hp2/ckpt-96/ckpt-222')


metric = dc.metrics.Metric(dc.metrics.r2_score, mode='regression')

num_epochs = 1
losses_second_train = []
score_second_valid = []
score_second_train = []
for i in range(num_epochs):
    loss_train = logD_model.fit(train_dataset, nb_epoch=1)
    R2_valid = logD_model.evaluate(valid_dataset,[metric])['r2_score']
    R2_train = logD_model.evaluate(train_dataset,[metric])['r2_score']
    print("Epoch %d loss_train: %f R2_train %f R2_valid: %f  " % (i, loss_train,R2_train,R2_valid))

    losses_second_train.append(loss_train)
    score_second_train.append(R2_train)
    score_second_valid.append(R2_valid)

###Real value and predicted value for target
train_y = train_dataset.y
train_pred = logD_model.predict(train_dataset)

test1_y = test1_dataset.y
test1_pred = logD_model.predict(test1_dataset)

test2_y = test2_dataset.y
test2_pred = logD_model.predict(test2_dataset)


###Show the results in figure
plt.figure(1)
plt.xlim((-4,4))
plt.ylim((-4,4))
plt.title("GCN Regression Prediction")
plt.xlabel("Real E (hartree)")
plt.ylabel("Predicted E (hartree)")
plt.grid(color='w', linestyle='--', linewidth=1)
plt.scatter(train_y,train_pred, 
            color="blue", alpha=0.8, label="train")
plt.scatter(test1_y,test1_pred, 
            color="red", alpha=0.8, label="test1")
plt.scatter(test2_y,test2_pred, 
            color="lightgreen", alpha=0.8, label="test2")
plt.legend(loc = 'best')
plt.show()


model = "DeepChem"
 # Scores of Train Data 
train_mae = mean_absolute_error(train_y, train_pred)
train_rmse = mean_squared_error(train_y, train_pred , squared=False)
train_r2 = r2_score(train_y, train_pred)
print('##########################  Scores of Train Data  ##########################')
print('Train set MAE of {}: {:.3f}'.format(model, train_mae))
print('Train set RMSE of {}: {:.3f}'.format(model, train_rmse))
print('Train set R2 Score of {}: {:.3f}'.format(model, train_r2))

print("----------------------------------------------------------------------------")

# Test1 Data
test1_mae = mean_absolute_error(test1_y,test1_pred)
test1_rmse = mean_squared_error(test1_y,test1_pred, squared=False)
test1_r2 = r2_score(test1_y,test1_pred)
print('##########################  Scores of Test1 Data  ##########################')
print('Test1 set MAE of {}: {:.3f}'.format(model, test1_mae))
print('Test1 set RMSE of {}: {:.3f}'.format(model, test1_rmse))
print('Test1 set R2 Score of {}: {:.3f}'.format(model, test1_r2))

print("----------------------------------------------------------------------------")

# Test2 Data
test2_mae = mean_absolute_error(test2_y,test2_pred)
test2_rmse = mean_squared_error(test2_y,test2_pred, squared=False)
test2_r2 = r2_score(test2_y,test2_pred)
print('##########################  Scores of Test2 Data  ##########################')
print('Test2 set MAE of {}: {:.3f}'.format(model, test2_mae))
print('Test2 set RMSE of {}: {:.3f}'.format(model, test2_rmse))
print('Test2 set R2 Score of {}: {:.3f}'.format(model, test2_r2))

print("----------------------------------------------------------------------------")





    