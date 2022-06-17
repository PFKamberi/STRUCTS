import numpy as np
import pandas as pd
import time
from scipy.io import arff
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from pyts.transformation import WEASEL
import minirocket
import matplotlib
import matplotlib.pyplot as plt
from progressbar import ProgressBar, AnimatedMarker, Bar
from progressbar import Bar, AdaptiveETA, Percentage, ProgressBar, SimpleProgress
from tabulate import tabulate

def train_test_prefix(X_training, X_test, Y_training, Y_test, method):
  
    training_time = 0
    test_time = 0

    if method == "minirocket":
    
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

        #print(classification_report(Y_test, Y_pred))
    
        return (Y_pred, accuracy_score(Y_test, Y_pred), f1_score(Y_test, Y_pred, average = 'weighted'), training_time, test_time)

    elif method == 'weasel':
    
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

    else:
        print("Unsupported method")
        return

def structs_univariate(training_data_path, test_data_path, method, n_splits = 5, dataset = None):

    data = pd.DataFrame(arff.loadarff(training_data_path)[0]).append(pd.DataFrame(arff.loadarff(test_data_path)[0]))
    X = data.iloc[:,:-1]
    Y = data.iloc[: ,-1]
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)  

    start = 0
    if method == 'minirocket':
        start = 9
    elif method == 'weasel':
        start = 11
    else:
        print("Unsupported method")
        return

    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros((n_splits,X.shape[1]+1-start))
    f1_scores = np.zeros((n_splits,X.shape[1]+1-start))

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
        for i in pbar(range(start,X.shape[1]+1)): 
            
            result = train_test_prefix(X_train[:,:i], X_test[:,:i], Y_train, Y_test, method = method)
            #returns (predictions, accuracy, f1 score, training time, test time)

            accuracies[fold][i-start] = result[1]
            f1_scores[fold][i-start] =  result[2]
            training_time[fold] = result[3]
            test_time[fold] = result[4]

        fold += 1

    training_time = training_time.mean()
    test_time = test_time.mean()

    accuracies_mean = accuracies.mean(axis=0)
    f1_scores_mean = f1_scores.mean(axis=0)

    earlinesses = np.array([float(x)/X.shape[1] for x in range(start, X.shape[1]+1)])

    harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)

    best_accuracy = accuracies_mean.max()
    best_accuracy_timepoint = np.argmax(accuracies_mean)

    best_f1_score = f1_scores_mean.max()
    best_f1_score_timepoint = np.argmax(f1_scores_mean)

    best_harmonic_mean = harmonic_means.max()
    best_harmonic_mean_timepoint = np.argmax(harmonic_means)

    print(tabulate([["Mean Training Time", training_time]], tablefmt="grid"))
    
    print(tabulate([['Accuracy', best_accuracy, str(best_accuracy_timepoint+start) + '/' + str(X.shape[1]), earlinesses[best_accuracy_timepoint], harmonic_means[best_accuracy_timepoint]], 
                    ['F1-score', best_accuracy, str(best_f1_score_timepoint+start) + '/' + str(X.shape[1]), earlinesses[best_f1_score_timepoint], harmonic_means[best_f1_score_timepoint]],
                    ['Harmonic Mean', best_harmonic_mean, str(best_harmonic_mean_timepoint+start) + '/' + str(X.shape[1]), earlinesses[best_harmonic_mean_timepoint], harmonic_means[best_harmonic_mean_timepoint]]                    
                   ], headers=['Metric', 'Best Value', 'Timepoint', 'Earliness', 'Harmonic Mean'], tablefmt="grid"))

    timepoints = np.arange(start,X.shape[1]+1) 

    #plot accuracy
    full_time_accuracy = np.repeat(accuracies_mean[-1], timepoints.shape[0] )
 
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints, full_time_accuracy , '--', color = 'blue', label = method + ' Full-Time')
    ax.plot(timepoints, accuracies_mean, color = 'blue', label = method + ' STRUCTS') 
    ax.plot([best_accuracy_timepoint+start], [best_accuracy], 'D', markersize = 10,  label = 'Best Accuracy')
    plt.xlabel("Truncation Timepoint")
    plt.ylabel("Accuracy")
    plt.ylim(0.0,1.0)
    plt.legend() 
    plt.show()

    #plot f1 score
    full_time_f1 = np.repeat(f1_scores_mean[-1], timepoints.shape[0] )
 
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints, full_time_f1 , '--', color = 'blue', label = method + ' Full-Time')
    ax.plot(timepoints, f1_scores_mean, color = 'blue', label = method + ' STRUCTS') 
    ax.plot([best_f1_score_timepoint+start], [best_f1_score], 'D', markersize = 10, label = 'Best F1-score')
    plt.xlabel("Truncation Timepoint")
    plt.ylabel("F1-score")
    plt.ylim(0.0,1.0)
    plt.legend() 
    plt.show()

    #plot harmonic mean
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints , harmonic_means , color='blue', label = method + ' STRUCTS') 
    ax.plot([best_harmonic_mean_timepoint+start], [best_harmonic_mean], 'D', markersize = 10,  label = 'Best Harmonic Mean')
    plt.ylabel("Harmonic Mean")
    plt.xlabel("Truncation Timepoint")
    plt.ylim(0.0,1.0)
    plt.legend() 
    plt.show()

    res = (accuracies_mean, (best_accuracy_timepoint+start, best_accuracy), 
            f1_scores_mean, (best_f1_score_timepoint+start, best_f1_score),
            earlinesses, harmonic_means,  
            training_time, test_time)
       
    if dataset != None:
      np.save(dataset + "_" + method + "_results.npy", np.array(res, dtype=object))

    return res

def structs_univariate_unfolded(X_training, X_test, Y_training, Y_test, method): 

    X_training = np.nan_to_num(X_training).astype(np.float32) # replace NaN with 0 
    X_test = np.nan_to_num(X_test).astype(np.float32) # replace NaN with 0 
    
    label_encoder = LabelEncoder()
    Y_training = label_encoder.fit_transform(Y_training)
    Y_test = label_encoder.fit_transform(Y_test)

    start = 0
    if method == 'minirocket':
        start = 9
    elif method == 'weasel':
        start = 11
    else:
        print("Unsupported method")
        return

    training_time = np.zeros(X_training.shape[1]+1-start)
    test_time = np.zeros(X_training.shape[1]+1-start)
    accuracies = np.zeros(X_training.shape[1]+1-start)
    f1_scores = np.zeros(X_training.shape[1]+1-start)
    preds = []

    pbar = ProgressBar(widgets=[ SimpleProgress(), ' ' ,  Percentage(), ' ', Bar(marker='-'),
               ' ', AdaptiveETA()])
    for i in pbar(range(start,X_training.shape[1]+1)):

        result = train_test_prefix(X_training[:,:i], X_test[:,:i], Y_training, Y_test, method = method)
        preds.append(result[0])
        #returns (predictions, accuracy, f1 score, training time, test time)

        accuracies[i-11] = result[1]
        f1_scores[i-11] = result[2]
        training_time[i-11] = result[3]
        test_time[i-11] = result[4]

    training_time = training_time.mean()
    test_time = test_time.mean()

        
    return (np.array(preds), training_time, test_time, accuracies, f1_scores)

def structs_multivariate(train_file_paths, test_file_paths, method, n_splits = 2, dataset = None):

    data = []
    for i in range(len(train_file_paths)):
        dim_data = pd.DataFrame(arff.loadarff(train_file_paths[i])[0]).append(pd.DataFrame(arff.loadarff(test_file_paths[i])[0])).values
        data.append(dim_data)
    data = np.array(data)
    data = np.transpose(data, (1, 0, 2))

    X = data[:,:,:-1]
    Y = data[:,:,-1]
   
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)  #random state?

    start = 0
    if method == 'minirocket':
        start = 9
    elif method == 'weasel':
        start = 11
    else:
        print("Unsupported method")
        return

    training_time = np.zeros(n_splits)
    test_time = np.zeros(n_splits)
    accuracies = np.zeros((n_splits, X.shape[2]+1-start)) 
    f1_scores  = np.zeros((n_splits, X.shape[2]+1-start))
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

            univarite_results = structs_univariate_unfolded(X_training[i], X_test[i], Y_training[i], Y_test[i], method = method)

            preds.append(univarite_results[0])
            training_time[fold] += univarite_results[1]
            test_time[fold] += univarite_results[2]

        
        preds = np.array(preds)
        preds_per_prefix = np.transpose(preds, (2, 1, 0))
        voted_Y_pred = []
        for item in preds_per_prefix:
            votes = []
            for x in item:
                votes.append(np.bincount(x).argmax())
            voted_Y_pred.append(votes)
        voted_Y_pred = np.array(voted_Y_pred)

        voted_Y_pred = voted_Y_pred.T 

        label_encoder = LabelEncoder()
        Y_test = label_encoder.fit_transform(Y_test.T[:,0])

        for i in range(voted_Y_pred.shape[0]): 
            accuracies[fold][i] = accuracy_score(voted_Y_pred[i], Y_test)
            f1_scores[fold][i]  = f1_score(voted_Y_pred[i], Y_test, average = 'weighted')
        
        fold += 1

    training_time = training_time.mean()
    test_time = test_time.mean()

    accuracies_mean = accuracies.mean(axis=0)

    f1_scores_mean = f1_scores.mean(axis=0)

    earlinesses = np.array([float(x)/X.shape[2] for x in range(start, X.shape[2]+1)])

    harmonic_means =  (2 * (1 - earlinesses) * accuracies_mean) / ((1 - earlinesses) + accuracies_mean)
    
    best_accuracy = accuracies_mean.max()
    best_accuracy_timepoint = np.argmax(accuracies_mean)

    best_f1_score = f1_scores_mean.max()
    best_f1_score_timepoint = np.argmax(f1_scores_mean)

    best_harmonic_mean = harmonic_means.max()
    best_harmonic_mean_timepoint = np.argmax(harmonic_means)


    print(tabulate([["Mean Training Time", training_time]], tablefmt="grid"))
    
    print(tabulate([['Accuracy', best_accuracy, str(best_accuracy_timepoint+start) + '/' + str(X.shape[2]), earlinesses[best_accuracy_timepoint], harmonic_means[best_accuracy_timepoint]], 
                    ['F1-score', best_accuracy, str(best_f1_score_timepoint+start) + '/' + str(X.shape[2]), earlinesses[best_f1_score_timepoint], harmonic_means[best_f1_score_timepoint]],
                    ['Harmonic Mean', best_harmonic_mean, str(best_harmonic_mean_timepoint+start) + '/' + str(X.shape[2]), earlinesses[best_harmonic_mean_timepoint], harmonic_means[best_harmonic_mean_timepoint]]                    
                   ], headers=['Metric', 'Best Value', 'Timepoint', 'Earliness', 'Harmonic Mean'], tablefmt="grid"))

    timepoints = np.arange(start,X.shape[2]+1) 

    #plot accuracy
    full_time_accuracy = np.repeat(accuracies_mean[-1], timepoints.shape[0] )
 
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints, full_time_accuracy , '--', color = 'blue', label = method + ' Full-Time')
    ax.plot(timepoints, accuracies_mean, color = 'blue', label = method + ' STRUCTS') 
    ax.plot([best_accuracy_timepoint+start], [best_accuracy], 'D', markersize = 10,  label = 'Best Accuracy')
    plt.xlabel("Truncation Timepoint")
    plt.ylabel("Accuracy")
    plt.ylim(0.0,1.0)
    plt.legend() 
    plt.show()

    #plot f1 score
    full_time_f1 = np.repeat(f1_scores_mean[-1], timepoints.shape[0] )
 
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints, full_time_f1 , '--', color = 'blue', label = method + ' Full-Time')
    ax.plot(timepoints, f1_scores_mean, color = 'blue', label = method + ' STRUCTS') 
    ax.plot([best_f1_score_timepoint+start], [best_f1_score], 'D', markersize = 10, label = 'Best F1-score')
    plt.xlabel("Truncation Timepoint")
    plt.ylabel("F1-score")
    plt.ylim(0.0,1.0)
    plt.legend() 
    plt.show()

    #plot harmonic mean
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.plot(timepoints , harmonic_means , color='blue', label = method + ' STRUCTS') 
    ax.plot([best_harmonic_mean_timepoint+start], [best_harmonic_mean], 'D', markersize = 10, label = 'Best Harmonic Mean')
    plt.ylabel("Harmonic Mean")
    plt.xlabel("Truncation Timepoint")
    plt.ylim(0.0,1.0)
    plt.legend() 
    plt.show()

    res = (accuracies_mean, (best_accuracy_timepoint+start, best_accuracy), 
            f1_scores_mean, (best_f1_score_timepoint+start, best_f1_score),
            earlinesses, harmonic_means,  
            training_time, test_time)
       
    if dataset != None:
      np.save(dataset + "_" + method + "_results.npy", np.array(res, dtype=object))

    return res
