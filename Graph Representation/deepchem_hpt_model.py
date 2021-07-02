#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:22:47 2021

@author: elham
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter,SpecifiedSplitter,RandomSplitter
from deepchem.models import GraphConvModel
import pickle
from keras.callbacks import ModelCheckpoint,EarlyStopping
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk
from deepchem.models.losses import Loss
from deepchem.utils.evaluate import Evaluator
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import time

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

model_dir = "./tf_chp_hp"
model = GraphConvModel(n_tasks=1, batch_size=32, mode='regression', dropout=0.0,dense_layer_size=256,learning_rate=0.005,model_dir= model_dir,random_seed=0)

metric = dc.metrics.Metric(dc.metrics.r2_score, mode='regression')

ckpt = tf.train.Checkpoint(step=tf.Variable(1))
manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=20)

start_time = time.time()

num_epochs = 100
losses_train = []
score_valid = []
score_train = []
for i in range(num_epochs):
    loss_train = model.fit(train_dataset, nb_epoch=1,deterministic=True)
    ckpt.step.assign_add(1)
    save_path = manager.save()
    print("Saved checkpoint for step {}: {} ".format(int(ckpt.step), save_path))
    model.save_checkpoint(max_checkpoints_to_keep=20 , model_dir = save_path )
    #model.restore()
    R2_train = model.evaluate(train_dataset,[metric])['r2_score']
    R2_valid = model.evaluate(valid_dataset,[metric])['r2_score']
    print("Epoch %d loss_train: %f R2_train %f R2_valid: %f  " % (i, loss_train,R2_train,R2_valid))
    
    
    losses_train.append(loss_train)
    score_valid.append(R2_valid)
    score_train.append(R2_train)
    

  
####Time 
print("--- %s seconds ---" % (time.time() - start_time)) 

df = pd.DataFrame(list(zip(losses_train,score_train,score_valid)),columns = ['train-loss','train-R2score','valid-R2score'])

df.to_csv("score-hpopt.csv")


#print(len(losses))
plt.figure(1)
fig, ax = plt.subplots(2, sharex='col', sharey='row')
x = range(num_epochs)
y_loss = losses_train
ax[0].plot(x, y_loss, c='b', alpha=0.6, label='loss_train')
ax[0].set(xlabel='epoch', ylabel='loss')

y_score = score_train
ax[1].plot(x, y_score,c='r', alpha=0.6, label='score_valid')
ax[1].set(xlabel='epoch', ylabel='R2 score')
   
    
###Real value and predicted value for target
train_y = train_dataset.y
train_pred = model.predict(train_dataset)

test1_y = test1_dataset.y
test1_pred = model.predict(test1_dataset)

test2_y = test2_dataset.y
test2_pred = model.predict(test2_dataset)


##evaluation model

dcmodel = "DeepChem"
 # Scores of Train Data 
train_mae = mean_absolute_error(dc.trans.undo_transforms(train_y,[transformers_train]), 
                                dc.trans.undo_transforms(train_pred,[transformers_train]))
train_rmse = mean_squared_error(dc.trans.undo_transforms(train_y,[transformers_train]), 
                                dc.trans.undo_transforms(train_pred,[transformers_train]) , squared=False)
train_r2 = r2_score(dc.trans.undo_transforms(train_y,[transformers_train]), 
                                dc.trans.undo_transforms(train_pred,[transformers_train]))
print('##########################  Scores of Train Data  ##########################')
print('Train set MAE of {}: {:.3f}'.format(dcmodel, train_mae))
print('Train set RMSE of {}: {:.3f}'.format(dcmodel, train_rmse))
print('Train set R2 Score of {}: {:.3f}'.format(dcmodel, train_r2))

print("----------------------------------------------------------------------------")

# Test1 Data
test1_mae = mean_absolute_error(dc.trans.undo_transforms(test1_y,[transformers_test1]),
                                dc.trans.undo_transforms(test1_pred,[transformers_test1]))
test1_rmse = mean_squared_error(dc.trans.undo_transforms(test1_y,[transformers_test1]),
                                dc.trans.undo_transforms(test1_pred,[transformers_test1]), squared=False)
test1_r2 = r2_score(dc.trans.undo_transforms(test1_y,[transformers_test1]),
                                dc.trans.undo_transforms(test1_pred,[transformers_test1]))
print('##########################  Scores of Test1 Data  ##########################')
print('Test1 set MAE of {}: {:.3f}'.format(dcmodel, test1_mae))
print('Test1 set RMSE of {}: {:.3f}'.format(dcmodel, test1_rmse))
print('Test1 set R2 Score of {}: {:.3f}'.format(dcmodel, test1_r2))

print("----------------------------------------------------------------------------")

# Test2 Data
test2_mae = mean_absolute_error(dc.trans.undo_transforms(test2_y,[transformers_test2]),
                                dc.trans.undo_transforms(test2_pred,[transformers_test2]))
test2_rmse = mean_squared_error(dc.trans.undo_transforms(test2_y,[transformers_test2]),
                                dc.trans.undo_transforms(test2_pred,[transformers_test2]), squared=False)
test2_r2 = r2_score(dc.trans.undo_transforms(test2_y,[transformers_test2]),
                                dc.trans.undo_transforms(test2_pred,[transformers_test2]))
print('##########################  Scores of Test2 Data  ##########################')
print('Test2 set MAE of {}: {:.3f}'.format(dcmodel, test2_mae))
print('Test2 set RMSE of {}: {:.3f}'.format(dcmodel, test2_rmse))
print('Test2 set R2 Score of {}: {:.3f}'.format(dcmodel, test2_r2))

print("----------------------------------------------------------------------------")



## evaluation using deepchem model metrics 

metric_mae = dc.metrics.Metric(dc.metrics.mean_absolute_error)
train_mae = model.evaluate(train_dataset, [metric_mae],[transformers_train])

test1_mae = model.evaluate(test1_dataset, [metric_mae],[transformers_test1])

test2_mae = model.evaluate(test2_dataset, [metric_mae],[transformers_test2])


metric_mse = dc.metrics.Metric(dc.metrics.mean_squared_error)

train_mse = model.evaluate(train_dataset, [metric_mse],[transformers_train])

test1_mse = model.evaluate(test1_dataset, [metric_mse],[transformers_test1])

test2_mse = model.evaluate(test2_dataset, [metric_mse],[transformers_test2])


metric_r2 = dc.metrics.Metric(dc.metrics.r2_score)
train_r2 = model.evaluate(train_dataset, [metric_r2],[transformers_train])

test1_r2 = model.evaluate(test1_dataset, [metric_r2],[transformers_test1])

test2_r2 = model.evaluate(test2_dataset, [metric_r2],[transformers_test2])


print("Train evaluation")
print(train_mae)
print(train_mse)
print(train_r2)

print("Test1 evaluation")
print(test1_mae)
print(test1_mse)
print(test1_r2)


print("Test2 evaluation")
print(test2_mae)
print(test2_mse)
print(test2_r2)

## plot for all train and test data
###Show the results in figure
plt.figure(1)
plt.xlim((-0.2,0.3))
plt.ylim((-0.2,0.2))
plt.title("GCN Regression Prediction")
plt.xlabel("Real E (hartree)")
plt.ylabel("Predicted E (hartree)")
plt.grid(color='w', linestyle='--', linewidth=1)
plt.scatter(dc.trans.undo_transforms(train_y,[transformers_train]),
                                dc.trans.undo_transforms(train_pred,[transformers_train]), 
            color="blue", alpha=0.8, label="train")
plt.scatter(dc.trans.undo_transforms(test1_y,[transformers_test1]),
                                dc.trans.undo_transforms(test1_pred,[transformers_test1]), 
            color="red", alpha=0.8, label="test1")
plt.scatter(dc.trans.undo_transforms(test2_y,[transformers_test2]),
                                dc.trans.undo_transforms(test2_pred,[transformers_test2]), 
            color="lightgreen", alpha=0.8, label="test2")
plt.legend(loc = 'best')
plt.show()
