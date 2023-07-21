import CC_functions as ccf
import numpy as np
import matplotlib.pyplot as plt
import os
import image_features
import pvml
import pandas as pd


#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°1.1 LOW LEVEL FEATURES EXTRACTION°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°


#EXTRACT LOW-LEVEL FEATURES (COLOR HISTOGRAMS) FROM IMAGES
classes = os.listdir("Data_gitignore/cake-classification/test")

X, Y = ccf.dir_feat_extract("Data_gitignore/cake-classification/test", classes, extract_method='CH')
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("Features/CH_test.txt.gz", data)

X, Y = ccf.dir_feat_extract("Data_gitignore/cake-classification/train", classes, extract_method='CH')
print("train", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("Features/CH_train.txt.gz", data)


#°°°°°°°°°°°°°°°°°°°°°°°°°°1.2 NEURAL FEATURES EXTRACTION°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°


#EXTRACT NEURAL FEATURES FROM IMAGES
classes = os.listdir("Data_gitignore/cake-classification/test")
cnn = pvml.CNN.load("Trained_models/pvmlnet.npz")

X, Y = ccf.dir_feat_neural("Data_gitignore/cake-classification/test", cnn, classes)
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("Features/neural_test.txt.gz", data)


X, Y = ccf.dir_feat_neural("Data_gitignore/cake-classification/train", cnn, classes)
print("train", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("Features/neural_train.txt.gz", data)


#°°°°°°°°°°°°°°°°°°°°°°°°2.1 COMBINING FEATURES: EXTRACT FEAT TO BE COMBINED°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°


#EXTRACT OTHERS LOW-LEVEL FEATURES FROM IMAGES
classes = os.listdir("Data_gitignore/cake-classification/test")

#EDGE DIRECTION HISTOGRAM
X, Y = ccf.dir_feat_extract("Data_gitignore/cake-classification/test", classes, extract_method='EDH')
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("Features/EDH_test.txt.gz", data)

X, Y = ccf.dir_feat_extract("Data_gitignore/cake-classification/train", classes, extract_method='EDH')
print("train", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("Features/EDH_train.txt.gz", data)


#COOCCURRENCE MATRIX
X, Y = ccf.dir_feat_extract("Data_gitignore/cake-classification/test", classes, extract_method='CM')
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("Features/CM_test.txt.gz", data)

X, Y = ccf.dir_feat_extract("Data_gitignore/cake-classification/train", classes, extract_method='CM')
print("train", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("Features/CM_train.txt.gz", data)


#°°°°°°°°°°°°°°°°°°°°°°°°°°°°2.1 COMBINED FEATURE EXTRACTION°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°


#load features to be combined
feature_names = ['CH_test', 'CH_train', 'EDH_test', 'EDH_train', 'CM_test', 'CM_train']
features_list = ccf.load_features(feature_names)
X_CH_test, Y_CH_test, X_CH_train, Y_CH_train, X_EDH_test, Y_EDH_test, X_EDH_train, Y_EDH_train, X_CM_test, Y_CM_test, X_CM_train, Y_CM_train = features_list

#HISTOGRAM + EDGE DIRECTION HISTOGRAM
X_test_CH_EDH, X_train_CH_EDH = ccf.concatenate_features([X_CH_test, X_EDH_test], Y_CH_test, [X_CH_train, X_EDH_train], Y_CH_train, 'CH_EDH_test', 'CH_EDH_train', store = True)

#HISTOGRAM + COOCCURRENCE MATRIX
X_test_CH_CM, X_train_CH_CM = ccf.concatenate_features([X_CH_test, X_CM_test], Y_CH_test, [X_CH_train, X_CM_train], Y_CH_train, 'CH_CM_test', 'CH_CM_train', store = True)

#EDGE DIRECTION HISTOGRAM + COOCCURRENCE MATRIX
X_test_EDH_CM, X_train_EDH_CM = ccf.concatenate_features([X_EDH_test, X_CM_test],  Y_CM_test, [X_EDH_train, X_CM_train], Y_CM_train, 'EDH_CM_test', 'EDH_CM_train', store = True)

#HISTOGRAM + EDGE DIRECTION HISTOGRAM + COOCCURRENCE MATRIX
X_test_CH_EDH_CM, X_train_CH_EDH_CM = ccf.concatenate_features([X_CH_test, X_EDH_test, X_CM_test], Y_CH_test, [X_CH_train, X_EDH_train, X_CM_train], Y_CH_train, 'CH_EDH_CM_test', 'CH_EDH_CM_train', store = True)


#°°°°°°°°°°°°°°°°°°°°°°°°°°2.3 NEURAL FEATURES EXTRACTION FROM DIFFERENT ACTIVATION LAYERS°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°


#use different activation layers as features

act_layers = [-1, -2, -3, -4, -5, -6, -7] #specify the layerS to extract features from
classes = os.listdir("Data_gitignore/cake-classification/test")
cnn = pvml.CNN.load("Trained_models/pvmlnet.npz")
    
for act_layer in act_layers:

    X, Y = ccf.dir_feat_neural("Data_gitignore/cake-classification/test", cnn, classes, act_layer)
    print("test", X.shape, Y.shape)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt(f"Features/neural{act_layer}_test.txt.gz", data)


    X, Y = ccf.dir_feat_neural("Data_gitignore/cake-classification/train", cnn, classes, act_layer)
    print("train", X.shape, Y.shape)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt(f"Features/neural{act_layer}_train.txt.gz", data)
