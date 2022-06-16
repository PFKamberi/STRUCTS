#Install the following packages if not already installed
#!pip install numba
#!pip install arff
#!pip install pyts

from numba import njit, prange, vectorize
import numpy as np
import argparse
import pandas as pd
import time
from scipy.io import arff
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from progressbar import ProgressBar, AnimatedMarker, Bar
from progressbar import Bar, AdaptiveETA, Percentage, ProgressBar, SimpleProgress
from sklearn.model_selection import StratifiedShuffleSplit
from pyts.transformation import WEASEL


####################### Minirocket transform #######################

@njit("float32[:](float32[:,:],int32[:],int32[:],float32[:])", fastmath = True, parallel = False, cache = True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles):

    num_examples, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype = np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            _X = X[np.random.randint(num_examples)]

            A = -_X          # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

    return biases

def _fit_dilations(input_length, num_features, max_dilations_per_kernel):

    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    
    dilations, num_features_per_dilation = \
    np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32) # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation

# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype = np.float32)

def fit(X, num_features = 10_000, max_dilations_per_kernel = 32):

    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)

    return dilations, num_features_per_dilation, biases

# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
@vectorize("float32(float32,float32)", nopython = True, cache = True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0

@njit("float32[:,:](float32[:,:],Tuple((int32[:],int32[:],float32[:])))", fastmath = True, parallel = False, cache = True) #was parallel = True
def transform(X, parameters):

    
    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases = parameters
    
    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)



    features = np.zeros((num_examples, num_features), dtype = np.float32)
    
    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation
            
            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()
                else:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()

                feature_index_start = feature_index_end
    return features

####################### Minirocket univariate ETSC #######################


#returns (predictions, accuracy, f1 score, training time, test time)
def minirocket_univariate_truncated(X_training, X_test, Y_training, Y_test):

    training_time = 0
    test_time = 0

    parameters = fit(X_training)

    # -- transform training ------------------------------------------------

    time_a = time.perf_counter()
    X_training_transform = transform(X_training, parameters)
    time_b = time.perf_counter()
    training_time += time_b - time_a

    # -- transform test ----------------------------------------------------

    time_a = time.perf_counter()
    X_test_transform = transform(X_test, parameters)
    time_b = time.perf_counter()
    test_time += time_b - time_a
        

    # -- training ----------------------------------------------------------

    time_a = time.perf_counter()
    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    classifier.fit(X_training_transform, Y_training)
    time_b = time.perf_counter()
    training_time += time_b - time_a

    # -- test --------------------------------------------------------------

    time_a = time.perf_counter()
    Y_pred = classifier.predict(X_test_transform)
    time_b = time.perf_counter()
    test_time += time_b - time_a
 
    return (Y_pred, accuracy_score(Y_test, Y_pred), f1_score(Y_test, Y_pred, average = 'weighted'), training_time, test_time)


def minirocket_univariate_brute_force_ETSC(training_data_path, test_data_path, n_splits = 5, dataset = None):

    data = pd.DataFrame(arff.loadarff(training_data_path)[0]).append(pd.DataFrame(arff.loadarff(test_data_path)[0]))
    X = data.iloc[:,:-1]
    Y = data.iloc[: ,-1]
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0) 
    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros((n_splits,X.shape[1]+1-9))
    f1_scores = np.zeros((n_splits,X.shape[1]+1-9))

    fold = 0
    for train_index, test_index in kfold.split(X,Y):
    

        X_training, X_testing = X.iloc[train_index], X.iloc[test_index]
        Y_training, Y_testing = Y.iloc[train_index], Y.iloc[test_index]

        X_train = np.nan_to_num(X_training).astype(np.float32) # replace NaN with 0 
        X_test = np.nan_to_num(X_testing).astype(np.float32)   # replace NaN with 0 

        label_encoder = LabelEncoder()
        Y_train = label_encoder.fit_transform(Y_training)
        Y_test = label_encoder.fit_transform(Y_testing)


        pbar = ProgressBar(widgets=['Fold: ' + str(fold+1)+ " | ", SimpleProgress(), ' ' ,  Percentage(), ' ', Bar(marker='-'),
               ' ', AdaptiveETA()])
        
        for i in pbar(range(9,X.shape[1]+1)): 
            

            result = minirocket_univariate_truncated(X_train[:,:i], X_test[:,:i], Y_train, Y_test)           
 
            accuracies[fold][i-9] = result[1]
            f1_scores[fold][i-9] =  result[2]
            training_time[fold] = result[3]
            test_time[fold] = result[4]

        fold += 1

   

    training_time = training_time.mean()
    test_time = test_time.mean()

    accuracies_mean = accuracies.mean(axis=0)
    accuracies_std = accuracies.std(axis=0)
    accuracies_ci = 0.1 * accuracies_std / accuracies_mean

    f1_scores_mean = f1_scores.mean(axis=0)
    f1_scores_std = f1_scores.std(axis=0)
    f1_scores_ci = 0.1 * f1_scores_std / f1_scores_mean

    accuracy_ci = 0.1 * accuracies_std /accuracies_mean

    earlinesses = np.array([float(x)/X.shape[1] for x in range(9, X.shape[1]+1)])

    harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)
    harmonic_means_f1 =  (2 * (1 - earlinesses) * f1_scores_mean) / ((1 - earlinesses) + f1_scores_mean)
    
    best_accuracy = accuracies_mean.max()
    best_accuracy_timepoint = np.argmax(accuracies_mean)

    best_f1_score = f1_scores_mean.max()
    best_f1_score_timepoint = np.argmax(f1_scores_mean)

    best_harmonic_mean = harmonic_means.max()
    best_harmonic_mean_timepoint = np.argmax(harmonic_means)


    print("\nTraining time mean: ", training_time)
    #print("Test time mean: ", test_time)

    print("Best accuracy: ", best_accuracy, " at timepoint ", best_accuracy_timepoint+9,'/', X.shape[1]) 
    print("Earliness at best accuracy: ", earlinesses[best_accuracy_timepoint]) 
    print("Harmonic mean at best accuracy: ", harmonic_means[best_accuracy_timepoint])
    
    print("Best f1 score: ", best_f1_score, " at timepoint ", best_f1_score_timepoint+9,'/', X.shape[1])
    print("Earliness at best f1 score: ", earlinesses[best_f1_score_timepoint])
    print("Harmonic mean at best f1 score: ", harmonic_means[best_f1_score_timepoint])

    print("Best harmonic mean: ", best_harmonic_mean, " at timepoint ", best_harmonic_mean_timepoint+9,'/',X.shape[1])

    res = (accuracies_mean, accuracies_ci, (best_accuracy_timepoint+9, best_accuracy), 
            f1_scores_mean, f1_scores_ci, (best_f1_score_timepoint+9, best_f1_score),
            earlinesses, harmonic_means, harmonic_means_f1, 
            training_time, test_time)
       
    if dataset != None:
      np.save(dataset+"_minirocket_results.npy", np.array(res, dtype=object))
    
    return res

####################### Weasel univariate ETSC #######################

def weasel_univariate_truncated(X_training, X_test, Y_training, Y_test, num_runs = 10):

    training_time = 0
    test_time = 0
    
    weasel = WEASEL(word_size=3, window_sizes =  [ 0.3, 0.4, 0.5,  0.6, 0.7, 0.8, 0.9], n_bins = 2)
    weasel.fit(X_training, Y_training)

    # -- transform training ------------------------------------------------

    time_a = time.perf_counter()
    X_training_transform = weasel.transform(X_training)
    time_b = time.perf_counter()
    training_time += time_b - time_a

    # -- transform test ----------------------------------------------------

    time_a = time.perf_counter()
    X_test_transform = weasel.transform(X_test)
    time_b = time.perf_counter()
    test_time += time_b - time_a
        

    # -- training ----------------------------------------------------------

    time_a = time.perf_counter()
    classifier = LogisticRegression(max_iter = 1000)
    classifier.fit(X_training_transform, Y_training)
    time_b = time.perf_counter()
    training_time += time_b - time_a

    # -- test --------------------------------------------------------------

    time_a = time.perf_counter()
    Y_pred = classifier.predict(X_test_transform)
    time_b = time.perf_counter()
    test_time += time_b - time_a
        
    return (Y_pred, accuracy_score(Y_test, Y_pred), f1_score(Y_test, Y_pred, average = 'weighted'), training_time, test_time)


def weasel_univariate_brute_force_ETSC(training_data_path, test_data_path, n_splits = 5, dataset = None):

    data = pd.DataFrame(arff.loadarff(training_data_path)[0]).append(pd.DataFrame(arff.loadarff(test_data_path)[0]))
    X = data.iloc[:,:-1]
    Y = data.iloc[: ,-1]
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)  


    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros((n_splits,X.shape[1]+1-11))
    f1_scores = np.zeros((n_splits,X.shape[1]+1-11))

    fold = 0
    
    for train_index, test_index in kfold.split(X,Y):
    

        X_training, X_testing = X.iloc[train_index], X.iloc[test_index]
        Y_training, Y_testing = Y.iloc[train_index], Y.iloc[test_index]

        X_train = np.nan_to_num(X_training).astype(np.float32) # replace NaN with 0 

        X_test = np.nan_to_num(X_testing).astype(np.float32)  # replace NaN with 0 

        label_encoder = LabelEncoder()
        
        Y_train = label_encoder.fit_transform(Y_training)
        Y_test = label_encoder.fit_transform(Y_testing)



        pbar = ProgressBar(widgets=['Fold: ' + str(fold+1)+ " | ", SimpleProgress(), ' ' ,  Percentage(), ' ', Bar(marker='-'),
               ' ', AdaptiveETA()])
        for i in pbar(range(11,X.shape[1]+1)): 
            
            result = weasel_univariate_truncated(X_train[:,:i], X_test[:,:i], Y_train, Y_test)
            #returns (predictions, accuracy, f1 score, training time, test time)

            accuracies[fold][i-11] = result[1]
            f1_scores[fold][i-11] =  result[2]
            training_time[fold] = result[3]
            test_time[fold] = result[4]

        fold += 1

    training_time = training_time.mean()
    test_time = test_time.mean()

    accuracies_mean = accuracies.mean(axis=0)
    accuracies_std = accuracies.std(axis=0)
    accuracies_ci = 0.1 * accuracies_std / accuracies_mean

    f1_scores_mean = f1_scores.mean(axis=0)
    f1_scores_std = f1_scores.std(axis=0)
    f1_scores_ci = 0.1 * f1_scores_std / f1_scores_mean

    accuracy_ci = 0.1 * accuracies_std /accuracies_mean

    earlinesses = np.array([float(x)/X.shape[1] for x in range(11, X.shape[1]+1)])

    harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)
    harmonic_means_f1 =  (2 * (1 - earlinesses) * f1_scores_mean) / ((1 - earlinesses) + f1_scores_mean)

    best_accuracy = accuracies_mean.max()
    best_accuracy_timepoint = np.argmax(accuracies_mean)

    best_f1_score = f1_scores_mean.max()
    best_f1_score_timepoint = np.argmax(f1_scores_mean)

    best_harmonic_mean = harmonic_means.max()
    best_harmonic_mean_timepoint = np.argmax(harmonic_means)


    print("\nTraining time mean: ", training_time)
    #print("Test time mean: ", test_time)

    print("Best accuracy: ", best_accuracy, " at timepoint ", best_accuracy_timepoint+11,'/', X.shape[1]) 
    print("Earliness at best accuracy: ", earlinesses[best_accuracy_timepoint]) 
    print("Harmonic mean at best accuracy: ", harmonic_means[best_accuracy_timepoint])
    
    print("Best f1 score: ", best_f1_score, " at timepoint ", best_f1_score_timepoint+11,'/', X.shape[1])
    print("Earliness at best f1 score: ", earlinesses[best_f1_score_timepoint])
    print("Harmonic mean at best f1 score: ", harmonic_means[best_f1_score_timepoint])

    print("Best harmonic mean: ", best_harmonic_mean, " at timepoint ", best_harmonic_mean_timepoint+11,'/',X.shape[1])

    res = (accuracies_mean, accuracies_ci, (best_accuracy_timepoint+11, best_accuracy), 
            f1_scores_mean, f1_scores_ci, (best_f1_score_timepoint+11, best_f1_score),
            earlinesses, harmonic_means, harmonic_means_f1, 
            training_time, test_time)
       
    if dataset != None:
      np.save(dataset+"_weasel_results.npy", np.array(res, dtype=object))
    
    return res

####################### Minirocket multivariate ETSC #######################
    
def minirocket_univariate_plain(X_training, X_test, Y_training, Y_test): #plain => without folds

    X_training = np.nan_to_num(X_training).astype(np.float32) # replace NaN with 0 
    X_test = np.nan_to_num(X_test).astype(np.float32) # replace NaN with 0 
    
    label_encoder = LabelEncoder()
    Y_training = label_encoder.fit_transform(Y_training)
    Y_test = label_encoder.fit_transform(Y_test)


    training_time = np.zeros(X_training.shape[1]+1-9)
    test_time = np.zeros(X_training.shape[1]+1-9)
    accuracies = np.zeros(X_training.shape[1]+1-9)
    f1_scores = np.zeros(X_training.shape[1]+1-9)
    preds = []

    pbar = ProgressBar(widgets=[ SimpleProgress(), ' ' ,  Percentage(), ' ', Bar(marker='-'),
               ' ', AdaptiveETA()])
    for i in pbar(range(9,X_training.shape[1]+1)):

        result = minirocket_univariate_truncated(X_training[:,:i], X_test[:,:i], Y_training, Y_test)
        preds.append(result[0])
        #returns (predictions, accuracy, f1 score, training time, test time)

        accuracies[i-9] = result[1]
        f1_scores[i-9] = result[2]
        training_time[i-9] = result[3]
        test_time[i-9] = result[4]

    training_time = training_time.mean()
    test_time = test_time.mean()

        
    return (np.array(preds), training_time, test_time, accuracies, f1_scores)

def minirocket_multivariate_brute_force_ETSC(train_file_paths, test_file_paths, n_splits = 2, dataset = None):

    data = []
    for i in range(len(train_file_paths)):
        dim_data = pd.DataFrame(arff.loadarff(train_file_paths[i])[0]).append(pd.DataFrame(arff.loadarff(test_file_paths[i])[0])).values
        data.append(dim_data)
    data = np.array(data)
    data = np.transpose(data, (1, 0, 2))

    X = data[:,:,:-1]
    Y = data[:,:,-1]

   
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)  #random state?

    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros((n_splits, X.shape[2]+1-9)) 
    f1_scores  = np.zeros((n_splits, X.shape[2]+1-9))
    _y_preds = []

    fold = 0
    for train_index, test_index in kfold.split(X,Y):
        print("\n\nFold: ", fold+1)
    
        X_training, X_test = X[train_index], X[test_index]
        Y_training, Y_test = Y[train_index], Y[test_index]

        X_training = np.transpose(X_training, (1, 0, 2))
        X_test  = np.transpose(X_test , (1, 0, 2))
        Y_training = np.transpose(Y_training)
        Y_test = np.transpose(Y_test)

        preds = []

        for i in range(len(train_file_paths)):

            print('\nDimension: ', i + 1)

            univarite_results = minirocket_univariate_plain(X_training[i], X_test[i], Y_training[i], Y_test[i])

            preds.append(univarite_results[0])
            training_time[fold] += univarite_results[1]
            test_time[fold] += univarite_results[2]

        
        preds = np.array(preds) #dim x length-9 x size
        preds_per_prefix = np.transpose(preds, (2, 1, 0)) # size x length - 9  x dims
        voted_Y_pred = []
        for item in preds_per_prefix:
            votes = []
            for x in item:
                votes.append(np.bincount(x).argmax())
            voted_Y_pred.append(votes)
        voted_Y_pred = np.array(voted_Y_pred) # size x length - 9

        voted_Y_pred = voted_Y_pred.T #length - 9 x size

        label_encoder = LabelEncoder()
        Y_test = label_encoder.fit_transform(Y_test.T[:,0])

        for i in range(voted_Y_pred.shape[0]): 
            accuracies[fold][i] = accuracy_score(voted_Y_pred[i], Y_test)
            f1_scores[fold][i]  = f1_score(voted_Y_pred[i], Y_test, average = 'weighted')
        

        fold += 1

    training_time = training_time.mean()
    test_time = test_time.mean()

    accuracies_mean = accuracies.mean(axis=0)
    accuracies_std = accuracies.std(axis=0)
    accuracies_ci = 0.1 * accuracies_std / accuracies_mean

    f1_scores_mean = f1_scores.mean(axis=0)
    f1_scores_std = f1_scores.std(axis=0)
    f1_scores_ci = 0.1 * f1_scores_std / f1_scores_mean


    earlinesses = np.array([float(x)/X.shape[2] for x in range(9, X.shape[2]+1)])

    harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)
    harmonic_means_f1 =  (2 * (1 - earlinesses) * f1_scores_mean) / ((1 - earlinesses) + f1_scores_mean)
    
    best_accuracy = accuracies_mean.max()
    best_accuracy_timepoint = np.argmax(accuracies_mean)

    best_f1_score = f1_scores_mean.max()
    best_f1_score_timepoint = np.argmax(f1_scores_mean)

    best_harmonic_mean = harmonic_means.max()
    best_harmonic_mean_timepoint = np.argmax(harmonic_means)


    print("\nTraining time mean: ", training_time)
    #print("Test time mean: ", test_time)

    print("Best accuracy: ", best_accuracy, " at timepoint ", best_accuracy_timepoint+9,'/', X.shape[2]) 
    print("Earliness at best accuracy: ", earlinesses[best_accuracy_timepoint]) 
    print("Harmonic mean at best accuracy: ", harmonic_means[best_accuracy_timepoint])
    
    print("Best f1 score: ", best_f1_score, " at timepoint ", best_f1_score_timepoint+9,'/', X.shape[2])
    print("Earliness at best f1 score: ", earlinesses[best_f1_score_timepoint])
    print("Harmonic mean at best f1 score: ", harmonic_means[best_f1_score_timepoint])

    print("Best harmonic mean: ", best_harmonic_mean, " at timepoint ", best_harmonic_mean_timepoint+9,'/',X.shape[2])

    res = (accuracies_mean, accuracies_ci, (best_accuracy_timepoint+9, best_accuracy), 
            f1_scores_mean, f1_scores_ci, (best_f1_score_timepoint+9, best_f1_score),
            earlinesses, harmonic_means, harmonic_means_f1, 
            training_time, test_time)
       
    if dataset != None:
      np.save(dataset+"_minirocket_results.npy", np.array(res, dtype=object))
    
    return res
    
####################### Weasel multivariate ETSC #######################

def weasel_univariate_plain(X_training, X_test, Y_training, Y_test): #plain => without folds

    X_training = np.nan_to_num(X_training).astype(np.float32) # replace NaN with 0 
    X_test = np.nan_to_num(X_test).astype(np.float32) # replace NaN with 0 
    
    label_encoder = LabelEncoder()
    Y_training = label_encoder.fit_transform(Y_training)
    Y_test = label_encoder.fit_transform(Y_test)


    training_time = np.zeros(X_training.shape[1]+1-11)
    test_time = np.zeros(X_training.shape[1]+1-11)
    accuracies = np.zeros(X_training.shape[1]+1-11)
    f1_scores = np.zeros(X_training.shape[1]+1-11)
    preds = []

    pbar = ProgressBar(widgets=[ SimpleProgress(), ' ' ,  Percentage(), ' ', Bar(marker='-'),
               ' ', AdaptiveETA()])
    for i in pbar(range(11,X_training.shape[1]+1)):

        result = weasel_univariate_truncated(X_training[:,:i], X_test[:,:i], Y_training, Y_test)
        preds.append(result[0])
        #returns (predictions, accuracy, f1 score, training time, test time)

        accuracies[i-11] = result[1]
        f1_scores[i-11] = result[2]
        training_time[i-11] = result[3]
        test_time[i-11] = result[4]

    training_time = training_time.mean()
    test_time = test_time.mean()

        
    return (np.array(preds), training_time, test_time, accuracies, f1_scores)

def weasel_multivariate_brute_force_ETSC(train_file_paths, test_file_paths,  n_splits = 2, dataset = None):

    data = []
    for i in range(len(train_file_paths)):
        dim_data = pd.DataFrame(arff.loadarff(train_file_paths[i])[0]).append(pd.DataFrame(arff.loadarff(test_file_paths[i])[0])).values
        '''X.append(dim_data[:,:-1])
        Y.append(dim_data[:,-1])'''
        data.append(dim_data)
    data = np.array(data)
    data = np.transpose(data, (1, 0, 2))

    X = data[:,:,:-1]
    Y = data[:,:,-1]
   
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)  #random state?

    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros((n_splits, X.shape[2]+1-11)) 
    f1_scores  = np.zeros((n_splits, X.shape[2]+1-11))
    _y_preds = []

    fold = 0
    for train_index, test_index in kfold.split(X,Y):
        print("\n\nFold: ", fold+1)
    
        X_training, X_test = X[train_index], X[test_index]
        Y_training, Y_test = Y[train_index], Y[test_index]

        X_training = np.transpose(X_training, (1, 0, 2))
        X_test  = np.transpose(X_test , (1, 0, 2))
        Y_training = np.transpose(Y_training)
        Y_test = np.transpose(Y_test)

        preds = []

        for i in range(len(train_file_paths)):

            print('\nDimension: ', i + 1)

            univarite_results = weasel_univariate_plain(X_training[i], X_test[i], Y_training[i], Y_test[i])

            preds.append(univarite_results[0])
            training_time[fold] += univarite_results[1]
            test_time[fold] += univarite_results[2]

        
        preds = np.array(preds) #dim x length-11 x size
        preds_per_prefix = np.transpose(preds, (2, 1, 0)) # size x length - 11  x dims
        voted_Y_pred = []
        for item in preds_per_prefix:
            votes = []
            for x in item:
                votes.append(np.bincount(x).argmax())
            voted_Y_pred.append(votes)
        voted_Y_pred = np.array(voted_Y_pred) # size x length - 11

        voted_Y_pred = voted_Y_pred.T #length - 11 x size

        label_encoder = LabelEncoder()
        Y_test = label_encoder.fit_transform(Y_test.T[:,0])

        for i in range(voted_Y_pred.shape[0]): 
            accuracies[fold][i] = accuracy_score(voted_Y_pred[i], Y_test)
            f1_scores[fold][i]  = f1_score(voted_Y_pred[i], Y_test, average = 'weighted')
        

        fold += 1

    training_time = training_time.mean()
    test_time = test_time.mean()

    accuracies_mean = accuracies.mean(axis=0)
    accuracies_std = accuracies.std(axis=0)
    accuracies_ci = 0.1 * accuracies_std / accuracies_mean

    f1_scores_mean = f1_scores.mean(axis=0)
    f1_scores_std = f1_scores.std(axis=0)
    f1_scores_ci = 0.1 * f1_scores_std / f1_scores_mean


    earlinesses = np.array([float(x)/X.shape[2] for x in range(11, X.shape[2]+1)])

    harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)
    harmonic_means_f1 =  (2 * (1 - earlinesses) * f1_scores_mean) / ((1 - earlinesses) + f1_scores_mean)
    
    best_accuracy = accuracies_mean.max()
    best_accuracy_timepoint = np.argmax(accuracies_mean)

    best_f1_score = f1_scores_mean.max()
    best_f1_score_timepoint = np.argmax(f1_scores_mean)

    best_harmonic_mean = harmonic_means.max()
    best_harmonic_mean_timepoint = np.argmax(harmonic_means)


    print("\nTraining time mean: ", training_time)
    #print("Test time mean: ", test_time)

    print("Best accuracy: ", best_accuracy, " at timepoint ", best_accuracy_timepoint+11,'/', X.shape[2]) 
    print("Earliness at best accuracy: ", earlinesses[best_accuracy_timepoint]) 
    print("Harmonic mean at best accuracy: ", harmonic_means[best_accuracy_timepoint])
    
    print("Best f1 score: ", best_f1_score, " at timepoint ", best_f1_score_timepoint+11,'/', X.shape[2])
    print("Earliness at best f1 score: ", earlinesses[best_f1_score_timepoint])
    print("Harmonic mean at best f1 score: ", harmonic_means[best_f1_score_timepoint])

    print("Best harmonic mean: ", best_harmonic_mean, " at timepoint ", best_harmonic_mean_timepoint+11,'/',X.shape[2])

    res = (accuracies_mean, accuracies_ci, (best_accuracy_timepoint+11, best_accuracy), 
            f1_scores_mean, f1_scores_ci, (best_f1_score_timepoint+11, best_f1_score),
            earlinesses, harmonic_means, harmonic_means_f1, 
            training_time, test_time)
       
    if dataset != None:
      np.save(dataset+"_weasel_results.npy", np.array(res, dtype=object))
    
    return res

# Univariate

#minirocket_ETSC_results = minirocket_univariate_brute_force_ETSC('./PowerCons_TRAIN.arff', './PowerCons_TEST.arff', n_splits = 5, dataset = 'PowerCons')
#weasel_ETSC_results = weasel_univariate_brute_force_ETSC('./PowerCons_TRAIN.arff', './PowerCons_TEST.arff', n_splits = 5, dataset = 'PowerCons')

#Multivariate

'''train_file_paths = ['/content/BasicMotionsDimension1_TRAIN.arff', '/content/BasicMotionsDimension2_TRAIN.arff',
                    '/content/BasicMotionsDimension3_TRAIN.arff', '/content/BasicMotionsDimension4_TRAIN.arff',
                    '/content/BasicMotionsDimension5_TRAIN.arff', '/content/BasicMotionsDimension6_TRAIN.arff']
test_file_paths = ['/content/BasicMotionsDimension1_TEST.arff', '/content/BasicMotionsDimension2_TEST.arff',
                    '/content/BasicMotionsDimension3_TEST.arff', '/content/BasicMotionsDimension4_TEST.arff',
                    '/content/BasicMotionsDimension5_TEST.arff', '/content/BasicMotionsDimension6_TEST.arff']'''

#minirocket_ETSC_results = minirocket_multivariate_brute_force_ETSC(train_file_paths, test_file_paths,  n_splits = 5, dataset = 'BasicMotions')
#weasel_ETSC_results = weasel_multivariate_brute_force_ETSC(train_file_paths, test_file_paths,  n_splits = 5, dataset = 'BasicMotions')
