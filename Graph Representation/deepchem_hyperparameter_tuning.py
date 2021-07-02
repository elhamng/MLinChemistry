#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 19:27:30 2021

@author: elham
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import deepchem as dc
from rdkit import Chem
from deepchem.splits.splitters import ScaffoldSplitter,SpecifiedSplitter,RandomSplitter
from deepchem.models.graph_models import GraphConvModel
from deepchem.utils.evaluate import Evaluator
from keras.callbacks import ModelCheckpoint
from deepchem.hyper import HyperparamOpt, GridHyperparamOpt, GaussianProcessHyperparamOpt
import os

####Read data from csv files , train, test1 and test2

input_data_train ="train_s.csv"




####Deepchem feature extraxtion and data loaded
tasks=['reaction_energy']
featurizer = dc.feat.ConvMolFeaturizer()
loader = dc.data.CSVLoader(tasks=tasks, feature_field="reactant_smiles",featurizer=featurizer)
dataset_train=loader.featurize(input_data_train)


##Normalized datasets
transformers_train = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset_train, move_mean=True)

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

metric = dc.metrics.Metric(dc.metrics.r2_score,task_averager=np.mean)


#######

def gc_model_builder(model_params , model_dir): 
    gc_model = dc.models.GraphConvModel(**model_params, model_dir = "models_new") 
    return gc_model

params_dict = {
   "np_epoch":[100],
    #"invalidparam!!!":[50],
    "n_tasks":[1],
    #"graph_conv_layers":[[64, 64]],
    "dense_layer_size":[64,128,256],
    "dropout":[0.0,0.1,0.2,0.3,0.4,0.5],
    "mode":["regression"],
    #"number_atom_features":[75],
    "batch_size": [32,64,128],
    "learning_rate": [0.005]#0.00001,0.0001,0.001,0.01,0.0005,0.005]
}

optimizer = dc.hyper.GridHyperparamOpt(lambda **p: dc.models.GraphConvModel(**p))


best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
    params_dict,
    train_dataset,
    valid_dataset,
    transformers = [transformers_train],
    metric = metric,
    use_max = False)

print("Best hyperparameters", best_hyperparams)
print("------------------------------------------")

print("All results")

print(all_results)