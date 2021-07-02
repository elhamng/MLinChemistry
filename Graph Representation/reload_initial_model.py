#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:58:04 2021

@author: elham
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import deepchem as dc
from rdkit import Chem
import matplotlib.pyplot as plt
from deepchem.splits.splitters import ScaffoldSplitter,SpecifiedSplitter,RandomSplitter
from deepchem.models import GraphConvModel
import pickle
from keras.callbacks import ModelCheckpoint,EarlyStopping
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk
from deepchem.utils.evaluate import Evaluator
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



####Read data from csv files , train, test1 and test2

input_data_train ="train_s.csv"
input_data_test1= '../final_data/test1.csv'
input_data_test2= '../final_data/test2.csv'
 

# Run before every test for reproducibility
def seed_all():
    np.random.seed(123)
    tf.random.set_seed(123)
    
    
# chose our targer and features
####Deepchem feature extraxtion and data loaded
tasks=['reaction_energy']
featurizer = dc.feat.ConvMolFeaturizer()
loader = dc.data.CSVLoader(tasks=tasks, feature_field="reactant_smiles",featurizer=featurizer)
dataset_train=loader.featurize(input_data_train)
dataset_test1=loader.featurize(input_data_test1)
dataset_test2=loader.featurize(input_data_test2)

##Normalized datasets
transformers_train = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_train, move_mean=True)
transformers_test1 = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_test1, move_mean=True)
transformers_test2 = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_test2,move_mean=True)


#There is a "split" field in the dataset file where I  defined the training/valid/test set
seed_all()
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset_train,frac_train = 0.8,frac_valid = 0.2, frac_test = 0.0,seed=0)

#Normalizes data set to zero mean
traint_dataset = transformers_train.transform(train_dataset)
validt_dataset= transformers_train.transform(valid_dataset)
test1_dataset = transformers_test1.transform(dataset_test1)
test2_dataset = transformers_test2.transform(dataset_test2)


model_dir = './tf_chp_initial'
logD_model = dc.models.GraphConvModel(n_tasks=1, batch_size=100, mode='regression', dropout=0.25,model_dir= model_dir,random_seed=0)
logD_model.restore('./tf_chp_initial/ckpt-100/ckpt-210')

metric = dc.metrics.Metric(dc.metrics.r2_score, mode='regression')


###Real value and predicted value for target
train_y = traint_dataset.y
train_pred = logD_model.predict(traint_dataset)

test1_y = test1_dataset.y
test1_pred = logD_model.predict(test1_dataset)

test2_y = test2_dataset.y
test2_pred = logD_model.predict(test2_dataset)


# ###Show the results in figure
# plt.figure(1)
# plt.xlim((-0.2,0.25))
# plt.ylim((-0.2,0.15))
# plt.title("GCN Regression Prediction")
# plt.xlabel("Real E (hartree)")
# plt.ylabel("Predicted E (hartree)")
# plt.grid(color='w', linestyle='--', linewidth=1)
# plt.scatter(dc.trans.undo_transforms(train_y,[transformers_train]),
#                                 dc.trans.undo_transforms(train_pred,[transformers_train]), 
#             color="blue", alpha=0.8, label="train")
# plt.scatter(dc.trans.undo_transforms(test1_y,[transformers_test1]),
#                                 dc.trans.undo_transforms(test1_pred,[transformers_test1]), 
#             color="red", alpha=0.8, label="test1")
# plt.scatter(dc.trans.undo_transforms(test2_y,[transformers_test2]),
#                                 dc.trans.undo_transforms(test2_pred,[transformers_test2]), 
#             color="lightgreen", alpha=0.8, label="test2")
# plt.legend(loc = 'best')
# plt.show()


# model = "DeepChem"
#   # Scores of Train Data 
# train_mae = mean_absolute_error(dc.trans.undo_transforms(train_y,[transformers_train]), 
#                                 dc.trans.undo_transforms(train_pred,[transformers_train]))
# print(train_mae)
# train_rmse = mean_squared_error(dc.trans.undo_transforms(train_y,[transformers_train]), 
#                                 dc.trans.undo_transforms(train_pred,[transformers_train]) , squared=False)
# train_r2 = r2_score(dc.trans.undo_transforms(train_y,[transformers_train]), 
#                                 dc.trans.undo_transforms(train_pred,[transformers_train]))
# print('##########################  Scores of Train Data  ##########################')
# print('Train set MAE of {}: {:.3f}'.format(model, train_mae))
# print('Train set RMSE of {}: {:.3f}'.format(model, train_rmse))
# print('Train set R2 Score of {}: {:.3f}'.format(model, train_r2))

# print("----------------------------------------------------------------------------")

# # Test1 Data
# test1_mae = mean_absolute_error(dc.trans.undo_transforms(test1_y,[transformers_test1]),
#                                 dc.trans.undo_transforms(test1_pred,[transformers_test1]))
# test1_rmse = mean_squared_error(dc.trans.undo_transforms(test1_y,[transformers_test1]),
#                                 dc.trans.undo_transforms(test1_pred,[transformers_test1]), squared=False)
# test1_r2 = r2_score(dc.trans.undo_transforms(test1_y,[transformers_test1]),
#                                 dc.trans.undo_transforms(test1_pred,[transformers_test1]))
# print('##########################  Scores of Test1 Data  ##########################')
# print('Test1 set MAE of {}: {:.3f}'.format(model, test1_mae))
# print('Test1 set RMSE of {}: {:.3f}'.format(model, test1_rmse))
# print('Test1 set R2 Score of {}: {:.3f}'.format(model, test1_r2))

# print("----------------------------------------------------------------------------")

# # Test2 Data
# test2_mae = mean_absolute_error(dc.trans.undo_transforms(test2_y,[transformers_test2]),
#                                 dc.trans.undo_transforms(test2_pred,[transformers_test2]))
# test2_rmse = mean_squared_error(dc.trans.undo_transforms(test2_y,[transformers_test2]),
#                                 dc.trans.undo_transforms(test2_pred,[transformers_test2]), squared=False)
# test2_r2 = r2_score(dc.trans.undo_transforms(test2_y,[transformers_test2]),
#                                 dc.trans.undo_transforms(test2_pred,[transformers_test2]))
# print('##########################  Scores of Test2 Data  ##########################')
# print('Test2 set MAE of {}: {:.3f}'.format(model, test2_mae))
# print('Test2 set RMSE of {}: {:.3f}'.format(model, test2_rmse))
# print('Test2 set R2 Score of {}: {:.3f}'.format(model, test2_r2))

# print("----------------------------------------------------------------------------")


# metric_mae = dc.metrics.Metric(dc.metrics.mean_absolute_error)
# train_mae = logD_model.evaluate(traint_dataset, [metric_mae],[transformers_train])

# test1_mae = logD_model.evaluate(test1_dataset, [metric_mae],[transformers_test1])

# test2_mae = logD_model.evaluate(test2_dataset, [metric_mae],[transformers_test2])


# metric_mse = dc.metrics.Metric(dc.metrics.mean_squared_error)

# train_mse = logD_model.evaluate(traint_dataset, [metric_mse],[transformers_train])

# test1_mse = logD_model.evaluate(test1_dataset, [metric_mse],[transformers_test1])

# test2_mse = logD_model.evaluate(test2_dataset, [metric_mse],[transformers_test2])


# metric_r2 = dc.metrics.Metric(dc.metrics.r2_score)
# train_r2 = logD_model.evaluate(traint_dataset, [metric_r2],[transformers_train])

# test1_r2 = logD_model.evaluate(test1_dataset, [metric_r2],[transformers_test1])

# test2_r2 = logD_model.evaluate(test2_dataset, [metric_r2],[transformers_test2])


# print("Train evaluation")
# print(train_mae)
# print(train_mse)
# print(train_r2)

# print("Test1 evaluation")
# print(test1_mae)
# print(test1_mse)
# print(test1_r2)


# print("Test2 evaluation")
# print(test2_mae)
# print(test2_mse)
# print(test2_r2)

### save data to csv files 
train_smile = train_dataset.ids
train_yo = dc.trans.undo_transforms(train_y,[transformers_train])
train_predo = dc.trans.undo_transforms(train_pred,[transformers_train])
#train_res = zip(train_smile,train_yo,train_predo)
df_train_pred = pd.DataFrame()
df_train_pred['smiles'] = train_smile
df_train_pred['train_y'] = train_yo
df_train_pred['train_pred'] = train_predo

test1_yo = dc.trans.undo_transforms(test1_y,[transformers_test1])
test1_predo = dc.trans.undo_transforms(test1_pred,[transformers_test1])
test1_smile = dataset_test1.ids
test1_res = zip(test1_smile,test1_yo,test1_predo)

df_test1_pred = pd.DataFrame()    
df_test1_pred['smiles'] = test1_smile
df_test1_pred['test1_y'] = test1_yo
df_test1_pred['test1_pred'] = test1_predo
    
                         

test2_yo = dc.trans.undo_transforms(test2_y,[transformers_test2])
test2_predo = dc.trans.undo_transforms(test2_pred,[transformers_test2])
test2_smile = dataset_test2.ids  
test2_res = zip(test2_smile,test2_yo,test2_predo)
df_test2_pred = pd.DataFrame()
df_test2_pred['smiles'] = test2_smile
df_test2_pred['test2_y'] = test2_yo
df_test2_pred['test2_pred'] = test2_predo
                      
 
df_train_pred.to_csv('init_pred_train.csv')
df_test1_pred.to_csv('init_pred_test1.csv')
df_test2_pred.to_csv('init_pred_test2.csv')
