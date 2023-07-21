import numpy as np
import matplotlib.pyplot as plt
import os
import image_features
import pvml
import pandas as pd


def train_MLP(mlp, X_train, Y_train, X_test, Y_test, n_epochs, batch, model_name, filename, store_model=True, store_accs = False, print_ = False):
    epochs = n_epochs
    batch_size = batch
    lr = 0.0001

    train_accs = []
    test_accs = []
    #plt.ion()
    for epoch in range(epochs):
        steps = X_train.shape[0] // batch_size
        mlp.train(X_train, Y_train, lr=lr, batch=batch_size, steps=steps, lambda_=0.001)
        if epoch % 100 == 0:
            predictions, probs = mlp.inference(X_train)
            train_acc = (predictions == Y_train).mean()
            train_accs.append(train_acc * 100)
            predictions, probs = mlp.inference(X_test)
            test_acc = (predictions == Y_test).mean()
            test_accs.append(test_acc * 100)
            if print_ is True:
                print(epoch, train_acc * 100, test_acc * 100)
            
    if store_accs is True:
        with open('Results/'+filename+'.csv', 'w') as f:
            print(f'epoch,train_acc,test_acc', file=f)
            for epoch, train_acc, test_acc in zip(range(epochs), train_accs, test_accs):
                print(f'{epoch},{train_acc},{test_acc}', file=f)
    
    if store_model is True:
        mlp.save('Trained_models/'+model_name+'.npz')
        
    return train_accs, test_accs


def extract_neural_features(im, net, act_layer):
    activations = net.forward(im[None, :, :, :])  # one more dimension, from 224x244x3 to 1x224x244x3
    features = activations[act_layer]
    #print(activations[act_layer].shape)
    features = features.reshape(-1)
    #print(features.shape)
    # for feature in features:
        # print(feature.shape)
    return features


def dir_feat_neural(path, net, classes, act_layer):
    all_features = []
    all_labels = []
    for klass_label, klass in enumerate(classes):
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path) / 255.0
            #print(image_path)
            features = extract_neural_features(image, net, act_layer)
            all_features.append(features)
            all_labels.append(klass_label)
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


def dir_feat_extract(path, classes, extract_method):
    '''Process a directory of images and creates a feature matrix and label vector for each feature extraction method in the extract_method'''
    
    classes = [c for c in classes if not c.startswith(".")] #remove hidden files (like .DS_Store)
    classes.sort()

    all_features = []
    all_labels = []
    klass_label = 0
    for klass in classes:
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path) / 255.0
            if extract_method == 'color_histogram' or extract_method == 'CH':
                features = image_features.color_histogram(image)
            elif extract_method == 'edge_direction_histogram' or extract_method == 'EDH':
                features = image_features.edge_direction_histogram(image)
            elif extract_method == 'cooccurrence_matrix' or extract_method == 'CM':
                features = image_features.cooccurrence_matrix(image)
            features = features.reshape(-1)
            all_features.append(features)
            all_labels.append(klass_label)
        klass_label += 1
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


def mean_var_normalize(train_features, test_features):
    '''returns normalized train and test features using mean-variance normalization'''
    u = train_features.mean(0)
    sigma = train_features.std(0)
    train_features = (train_features - u) / sigma
    test_features = (test_features - u) / sigma

    return train_features, test_features


def min_max_normalize(train_features, test_features):
    '''returns normalized train and test features using max-min normalization'''
    min = train_features.min(0)
    max = train_features.max(0)
    train_features = (train_features - min) / (max - min)
    test_features = (test_features - min) / (max - min)
    
    return train_features, test_features
    
    
def max_abs_normalize(train_features, test_features):
    '''returns normalized train and test features using max-abs normalization'''
    max = np.abs(train_features).max(0)
    train_features = train_features / max
    test_features = test_features / max
    
    return train_features, test_features


def max_abs_normalize(train_features, test_features):
    '''returns normalized train and test features using max-abs normalization'''
    max = np.abs(train_features).max(0)
    train_features = train_features / max
    test_features = test_features / max
    
    return train_features, test_features


def whitening_normalize(Xtrain , Xtest):
    mu = Xtrain.mean(0)
    sigma = np.cov(Xtrain.T)
    evals , evecs = np.linalg.eigh(sigma) 
    w = evecs / np.sqrt(evals)
    Xtrain = (Xtrain - mu) @ w 
    Xtest = (Xtest - mu) @ w 
    return Xtrain , Xtest


def confusion_matrix(Y, predictions, labels, show = False, rnorm = True):
    '''Displays the confusion matrix. If rnorm is True, the values are normalized respect to the actual number of samples in each class'''
    classes = Y.max() + 1
    cm = np.empty((classes, classes))
    for klass in range(classes):
        sel = (Y == klass).nonzero()
        counts = np.bincount(predictions[sel], minlength=classes)
        if rnorm is False:
            cm[klass, :] = counts
        else:
            cm[klass, :] = 100 * counts / max(1, counts.sum())
        
    if show is True:
        plt.figure(3, figsize=(20, 20))
        plt.clf()
        plt.xticks(range(classes), labels, rotation=45)
        plt.yticks(range(classes), labels)
        plt.imshow(cm, vmin=0, vmax=100, cmap='Blues')
        for i in range(classes):
            for j in range(classes):
                txt = "{:.1f}".format(cm[i, j], ha="center", va="center")
                col = ("black" if cm[i, j] < 75 else "white")
                plt.text(j - 0.25, i, txt, color=col)
        plt.title("Confusion Matrix") 
    return cm


def likely_misclassified(occurrs, cm, classes, n, show = False):
    '''Returns the n classes for which the delta between 100 and the percentage of correct classified is the largest'''
    deltas = []
    for i in range(15):
        delta = 100-cm[i,i]
        deltas.append(delta)
    indexes = np.argsort(deltas)
    deltas = np.array(deltas)
    indexes = indexes[-n:]
    arr = np.array([classes[indexes], np.floor(deltas[indexes]), occurrs[indexes]])
    if show is True:
        plt.figure(1)
        plt.vlines(arr[0,:], ymin = 0, ymax = arr[1,:])
        #plt.gca().invert_yaxis()
        plt.title('Most Likely Misclassified')
        plt.ylabel('Delta: 100 - correct (%)')
        plt.xticks(rotation = 90)
        plt.show(1)
    return arr


def wrong_classes(cm, classes):
    '''For each class returns the class respect to which it is more likely to be exchanged (with labels and with values)'''
    actuals = [] 
    wrongs = []
    vals = []
    for i in range(35):
        c = np.argmax(cm[i,:])
        if c == i:
            cm[i,c]=0
            c = np.argmax(cm[i,:])
        value = cm[i,c]
        actuals.append(i)
        wrongs.append(c)
        vals.append(value)
    out_valued = np.array([actuals, wrongs, vals])
    out_labelled = np.array([classes[actuals], classes[wrongs]])
    return out_labelled, out_valued


def wrong_classes(cm, classes):
    '''For each class returns the class respect to which it is more likely to be exchanged (with labels and with values)'''
    actuals = [] 
    wrongs = []
    vals = []
    for i in range(15):
        c = np.argmax(cm[i,:])
        if c == i:
            cm[i,c]=0
            c = np.argmax(cm[i,:])
        value = cm[i,c]
        actuals.append(i)
        wrongs.append(c)
        vals.append(value)
    out_valued = np.array([actuals, wrongs, vals])
    out_labelled = np.array([classes[actuals], classes[wrongs]])
    return out_labelled, out_valued


def item_prediction(imagename, classes, network, show = False):
    '''takes an image, the classes and a network and returns the first 5 most likely classes for the image according to the network'''
    
    imagepath = 'Data_gitignore/cake-classification/test/'+imagename

    image = plt.imread(imagepath) / 255.0
    labels, probs = network.inference(image[None, :, :, :])

    classes = [c for c in classes if not c.startswith(".")]
    classes.sort()

    c_list = []
    p_list=[]
    indices = (-probs[0]).argsort()
    for k in range(5):
        index = indices[k]
        c_list.append(classes[index])  
        p_list.append(probs[0][index] * 100)
    
    if show is True:
        plt.bar(x = c_list, height = p_list)
        plt.title(imagename)
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
    
    return c_list, p_list


def load_features(features_names):
    '''loads the features from the path'''
    
    list_of_features = []
    for name in features_names:
        features = np.loadtxt('Features/' + name + '.txt.gz')
        X = features[:, :-1]
        list_of_features.append(X)
        Y = features[:, -1].astype(int)
        list_of_features.append(Y)
    
    return list_of_features


def concatenate_features(X_test_list, Y_test, X_train_list, Y_train, filename_test, filename_train, store = True):
    '''concatenates the features'''
    
    X_test = np.concatenate(X_test_list, 1)
    data_test = np.concatenate([X_test, Y_test[:, None]], 1)

    X_train = np.concatenate(X_train_list, 1)
    data_train = np.concatenate([X_train, Y_train[:, None]], 1)
    
    if store is True:
        np.savetxt(f"Features/{filename_test}_test.txt.gz", data_test)
        np.savetxt(f"Features/{filename_train}_train.txt.gz", data_train)
    
    return data_test, data_train
        