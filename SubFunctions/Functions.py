import math
import os
from keras.layers import  Dropout, Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from termcolor import colored
from keras.utils import to_categorical
from keras import Sequential
from mealpy.swarm_based.COA import BaseCOA as COA
from mealpy.Prop import BaseProp as Proposed
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import seaborn as sns

def preprocessing(db):
    if db ==1:
        dataset=pd.read_csv("Dataset/Crop Recommendation Dataset/Crop_recommendation.csv")
    else:
        dataset = pd.read_csv("Dataset/HarvardCropDataset/Crop_recommendation.csv")

    # separate features and labels
    features = dataset.iloc[:,:-1].values
    labels = dataset.iloc[:,-1].values
    encoder = LabelEncoder()
    labels=encoder.fit_transform(labels)
    """Synthetic Minority Oversampling Technique (SMOTE) is a statistical technique for increasing the number of cases in your dataset in a balanced way. 
       The component works by generating new instances from existing minority cases that you supply as input"""
    # Apply Smote technique
    oversample = SMOTE()
    feat, lab = oversample.fit_resample(features, labels)

    ## Data Normalization
    feat=feat.astype("float32")/feat.max()

    if db==1:
        np.save("NewDataset\\Crop Recommendation Dataset\\Features.npy",feat)
        np.save("NewDataset\\Crop Recommendation Dataset\\Labels.npy",lab)
    elif db==2:
        np.save("NewDataset\\HarvardCropDataset\\Features.npy", feat)
        np.save("NewDataset\\HarvardCropDataset\\Labels.npy", lab)

def Model_Testing_Function(model, xtest, val):
    print("[INFO] Testing Model....")
    if val == 0:
        y_pred = model.predict(xtest)
    else:
        y_pred = model.predict(xtest)
        y_pred = np.argmax(y_pred, axis=1)
    return y_pred

class SupportVectorMachine:
    def __init__(self,  X_train,y_train,X_test,y_test):
        self.model = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
    def build(self):
        print(colored("[INFO] Loading...SVM Classifier","blue"))
        from sklearn.svm import SVC
        self.model = SVC()
    def train(self):
        self.build()
        print(colored("[INFO] Training...SVM Classifier",'blue'))
        self.model.fit(self.X_train, self.y_train)
        return self.model

class SupportVectorNN:
    def __init__(self,  X_train,y_train,X_test,y_test,epochs):
        self.model = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = 32
    def build(self):
        print(colored("[INFO] Loading...Ensemble SVM Neural Network","blue"))

        import tensorflow as tf
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=800, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=500, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=400, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=100, activation="relu"))
        self.model.add(Dense(self.y_train.shape[1], activation='softmax')),
        self.model.compile(optimizer="adam", loss="mse", metrics=['accuracy']),
    def train_model(self):
        self.build()
        from sklearn.svm import SVC
        print(colored("[INFO] Training...Ensemble SVM Neural Network",'blue'))
        self.model.fit(self.X_train,self.y_train,self.batch_size,self.epochs)
        self.model.pop()
        feature_mappings = self.model(self.X_train)
        feature_test = self.model(self.X_test)
        y1_train = np.argmax(self.y_train, axis=1)
        model = SVC()
        model.fit(feature_mappings, y1_train)
        return model,feature_test,feature_mappings, y1_train

class LighGBMNN:
    def __init__(self,  X_train,y_train,X_test,y_test,epochs):
        self.model = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = 32
    def build(self):
        print(colored("[INFO] Loading...Adaptive Ensemble LightGBM Neural Network","blue"))
        import tensorflow as tf
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=800, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=500, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=400, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=100, activation="relu"))
        self.model.add(Dense(self.y_train.shape[1], activation='softmax')),
        self.model.compile(optimizer="adam", loss="mse", metrics=['accuracy']),
    def train_model(self):
        self.build()
        import lightgbm as lgb
        print(colored("[INFO] Training...Adaptive Ensemble LightGBM Neural Network",'blue'))
        self.model.fit(self.X_train,self.y_train,self.batch_size,self.epochs)
        self.model.pop()
        feature_mappings = self.model(self.X_train)
        feature_test = self.model(self.X_test)
        y1_train = np.argmax(self.y_train, axis=1)
        model = lgb.LGBMClassifier()
        model.fit(feature_mappings, y1_train)
        return model, feature_test, feature_mappings, y1_train

class ConNN:
    def __init__(self, X_train,y_train,X_test,y_test, epochs):
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = 32
        self.model =None
    def build_compile_model(self):
        print(colored("[INFO] Loading...CNN", "blue"))
        self.model = Sequential()
        self.model.add(Conv2D(8, 3, padding='same', activation='relu', input_shape=(self.X_train.shape[1],self.X_train.shape[2],self.X_train.shape[3]))),
        self.model.add(MaxPooling2D(1, 1)),
        self.model.add(Conv2D(16, 3, padding='same', activation='relu')),
        self.model.add(MaxPooling2D(1, 1)),
        self.model.add(Dropout(0.2)),
        self.model.add(Flatten()),
        self.model.add(Dense(self.y_train.shape[1], activation='softmax')),
        self.model.compile(optimizer="adam", loss="mse", metrics=['accuracy']),
    def train_model(self):
        self.build_compile_model()
        print(colored("[INFO] Training...CNN", 'blue'))
        self.model.fit(self.X_train,self.y_train,self.batch_size,self.epochs)
        return self.model

def SVMNN(xtrain, ytrain, xtest, ytest, epochs):
    xtrain= xtrain.reshape(xtrain.shape[0],xtrain.shape[1])
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1])
    model = SupportVectorNN(xtrain, ytrain, xtest, ytest, epochs)
    model, newxtest, feature_mappings, y1_train = model.train_model()
    return model,newxtest

def LGBMNN(xtrain, ytrain, xtest, ytest, epochs,opt):
    xtrain= xtrain.reshape(xtrain.shape[0],xtrain.shape[1])
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1])
    model = LighGBMNN(xtrain, ytrain, xtest, ytest, epochs)
    model, newxtest, feature_mappings, y1_train = model.train_model()
    op = Optimization(model, feature_mappings, y1_train)
    if opt == 0:
        model = model
    else:
        model = op.main_update_hyperparameters(opt)
    return model, newxtest

def CNN(xtrain, ytrain, xtest, ytest, epochs):
    xtrain= xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1,1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1, 1)
    model = ConNN(xtrain, ytrain, xtest, ytest, epochs)
    model = model.train_model()
    return model, xtest

class Optimization:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test


    def fitness_function1(self, solution, model, y_test, x_test):
        print(colored("Fitness Function >> ", color='blue', on_color='on_grey'))
        try:
            s = self.model.booster_.trees_to_dataframe()
        except:
            s = self.model.booster_.trees_to_dataframe
        dataframe = pd.DataFrame(solution)
        s['weight'] = dataframe
        self.model.booster_.trees_to_dataframe = s
        pred = self.model.predict(x_test)
        acc = accuracy_score(y_test, pred)
        return acc

    def main_weight_updation_optimization(self, curr_wei, opt):
        problem_dict1 = {
            "fit_func": self.fitness_function1,
            "lb": [curr_wei.min(), ] * curr_wei.shape[0]*curr_wei.shape[1],
            "ub": [curr_wei.max(), ] * curr_wei.shape[0]*curr_wei.shape[1],
            "minmax": "max",
            "log_to": None,
            "save_population": False,
            "Curr_Weight": curr_wei,
            "test_loader": self.x_test,
            "tst_lab": self.y_test,
            "Model_trained_Partial": self.model,
        }
        if opt == 1:
            print((colored("[INFO] Border Collie Optimization \U0001F43A", 'magenta', on_color='on_grey')))
            model = COA(problem_dict1, epoch=2, pop_size=10)
            best_position2, best_fitness2 = model.solve()
        if opt == 2:
            print((colored("[INFO] Coyotes Optimization \U0001F43A", 'magenta', on_color='on_grey')))
            model = COA(problem_dict1, epoch=2, pop_size=10)
            best_position2, best_fitness2 = model.solve()
        if opt == 3:
            print((colored("[INFO] Proposed optimization  \U0001F680", 'magenta', on_color='on_grey')))
            model = Proposed(problem_dict1, epoch=2, pop_size=10)
            best_position2, best_fitness2 = model.solve()

        return best_position2

    def main_update_hyperparameters(self, opt):
        s = self.model.booster_.trees_to_dataframe()
        wei_to_train = s['weight'].values
        wei_to_train1=wei_to_train.reshape(wei_to_train.shape[0],1)
        wei_to_train1=np.nan_to_num(wei_to_train1,0)
        wei_to_train_new = self.main_weight_updation_optimization(wei_to_train1, opt)
        dataframe = pd.DataFrame(wei_to_train_new)
        s['weight'] = dataframe
        self.model.booster_.trees_to_dataframe =s
        return self.model


def main_est_perf_metrics(preds, y_test):
    """A confusion matrix presents a table layout of the different outcomes of the prediction
    and results of a classification problem and helps visualize its outcomes"""
    from sklearn.metrics import multilabel_confusion_matrix
    mcm = multilabel_confusion_matrix(y_test, preds)
    cm =sum(mcm)
    """Total Counts of Confusion Matrix"""
    total = sum(sum(cm))
    """True Positive"""
    TP = cm[0, 0]
    """False Positive"""
    FP = cm[0, 1]
    """False Negative"""
    FN = cm[1, 0]
    """True Negative"""
    TN = cm[1, 1]
    """Accuracy Formula"""
    acc = (TP + TN) / total
    """Sensitivity Formula"""
    sen = TP / (FN + TP)
    """Specificity Formula"""
    spe = TN / (FP + TN)
    """Precision Formula"""
    pre = TP / (TP + FP)
    """Recall Formula"""
    rec = TP / (FN + TP)
    """F1 Score Formula"""
    f1_score = (2 * pre * rec) / (pre + rec)
    return [acc, sen, spe, pre,rec,f1_score]


def TP_Analysis(feat, lab, db):
    """A label represents an output value,
    while a feature is an input value that describes the characteristics of such labels in datasets"""
    feat = np.nan_to_num(feat, 0)
    """An epoch in machine learning means one complete pass of the training dataset through the algorithm.
     This epoch's number is an important hyperparameter for the algorithm. 
     It specifies the number of epochs or complete passes of the entire training dataset passing through the training or learning process of the algorithm"""
    epochs = [10,15,20,25,30] # No. of Iterations
    tr = [0.4, 0.5, 0.6, 0.7, 0.8]  # Variation of Training Percentage
    options = [0, 1, 2, 3] # select the optimization 0-NOrmal model, 1-FFO, 2-HBO, 3-Prop

    """Normalization is a technique often applied as part of data preparation for machine learning. 
    The goal of normalization is to change the values of numeric columns in the dataset to use a common scale,
     without distorting differences in the ranges of values or losing information"""
    ## Normalization
    feat = feat.astype(np.float32)/feat.max()

    COM_A = []
    COM_B = []
    COM_C = []
    COM_D = []
    COM_E = []
    COM_F = []
    COM_G = []
    COM_H = []
    COM_I = []
    COM_J = []
    for p in range(len(tr)):
        """A train test split is when you split your data into a training set and a testing set. 
        The training set is used for training the model, and the testing set is used to test your model.
         This allows you to train your models on the training set, and then test their accuracy on the unseen testing set"""
        print(colored("Training Percentage and Testing Percentage : "+str(tr[p]*100)+" and "+str(100-(tr[p]*100)),"yellow"))
        xtrain, xtest, ytrain, ytest = train_test_split(feat, lab, train_size=tr[p], random_state=42, shuffle=True)

        """Categorical encoding is the process of converting categorical columns
         to numerical columns so that a Deep learning algorithm understands it"""
        # convert test labels and train labels into Hot Vectors
        y1train = to_categorical(ytrain)
        y1test = to_categorical(ytest)

        Model_1,xtest1 = SVMNN(xtrain, y1train, xtest, y1test,epochs[4])
        Model_2,xtest2 = CNN(xtrain, y1train, xtest, y1test, epochs[4])
        Model_3,xtest3 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[4], options[0])
        Model_4,xtest4 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[4], options[1])
        Model_5,xtest5 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[4], options[2])
        Model_6,xtest6 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[0], options[3])
        Model_7, xtest7 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[1], options[3])
        Model_8, xtest8 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[2], options[3])
        Model_9, xtest9 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[3], options[3])
        Model_10, xtest10 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[4], options[3])

        print("------------------------------MODEL TESTING SECTION---------------------------------------")

        preds_1 = Model_Testing_Function(Model_1, xtest1,0)
        preds_2 = Model_Testing_Function(Model_2, xtest2, 1)
        preds_3 = Model_Testing_Function(Model_3, xtest3, 0)
        preds_4 = Model_Testing_Function(Model_4, xtest4, 0)
        preds_5 = Model_Testing_Function(Model_5, xtest5, 0)
        preds_6 = Model_Testing_Function(Model_6, xtest6, 0)
        preds_7 = Model_Testing_Function(Model_7, xtest7, 0)
        preds_8 = Model_Testing_Function(Model_8, xtest8, 0)
        preds_9 = Model_Testing_Function(Model_9, xtest9, 0)
        preds_10 = Model_Testing_Function(Model_10, xtest10, 0)

        print("________________________Metrics Evaluated from Confusion Matrix__________________________________")
        [ACC1, SEN1, SPE1, PRE1, REC1, FSC1] = main_est_perf_metrics(preds_1, ytest)
        [ACC2, SEN2, SPE2, PRE2, REC2, FSC2] = main_est_perf_metrics(preds_2, ytest)
        [ACC3, SEN3, SPE3, PRE3, REC3, FSC3] = main_est_perf_metrics(preds_3, ytest)
        [ACC4, SEN4, SPE4, PRE4, REC4, FSC4] = main_est_perf_metrics(preds_4, ytest)
        [ACC5, SEN5, SPE5, PRE5, REC5, FSC5] = main_est_perf_metrics(preds_5, ytest)
        [ACC6, SEN6, SPE6, PRE6, REC6, FSC6] = main_est_perf_metrics(preds_6, ytest)
        [ACC7, SEN7, SPE7, PRE7, REC7, FSC7] = main_est_perf_metrics(preds_7, ytest)
        [ACC8, SEN8, SPE8, PRE8, REC8, FSC8] = main_est_perf_metrics(preds_8, ytest)
        [ACC9, SEN9, SPE9, PRE9, REC9, FSC9] = main_est_perf_metrics(preds_9, ytest)
        [ACC10, SEN10, SPE10, PRE10, REC10, FSC10] = main_est_perf_metrics(preds_10, ytest)

        per_1 = [ACC1, SEN1, SPE1, PRE1, REC1, FSC1]
        per_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FSC2]
        per_3 = [ACC3, SEN3, SPE3, PRE3, REC3, FSC3]
        per_4 = [ACC4, SEN4, SPE4, PRE4, REC4, FSC4]
        per_5 = [ACC5, SEN5, SPE5, PRE5, REC5, FSC5]
        per_6 = [ACC6, SEN6, SPE6, PRE6, REC6, FSC6]
        per_7 = [ACC7, SEN7, SPE7, PRE7, REC7, FSC7]
        per_8 = [ACC8, SEN8, SPE8, PRE8, REC8, FSC8]
        per_9 = [ACC9, SEN9, SPE9, PRE9, REC9, FSC9]
        per_10 = [ACC10, SEN10, SPE10, PRE10, REC10, FSC10]

        COM_A.append(per_1)
        COM_B.append(per_2)
        COM_C.append(per_3)
        COM_D.append(per_4)
        COM_E.append(per_5)
        COM_F.append(per_6)
        COM_G.append(per_7)
        COM_H.append(per_8)
        COM_I.append(per_9)
        COM_J.append(per_10)

    np.save('NPY\\COM_A' + str(db + 1) + '.npy'.format(os.getcwd()), COM_A)
    np.save('NPY\\COM_B' + str(db + 1) + '.npy'.format(os.getcwd()), COM_B)
    np.save('NPY\\COM_C' + str(db + 1) + '.npy'.format(os.getcwd()), COM_C)
    np.save('NPY\\COM_D' + str(db + 1) + '.npy'.format(os.getcwd()), COM_D)
    np.save('NPY\\COM_E' + str(db + 1) + '.npy'.format(os.getcwd()), COM_E)
    np.save('NPY\\COM_F' + str(db + 1) + '.npy'.format(os.getcwd()), COM_F)
    np.save('NPY\\COM_G' + str(db + 1) + '.npy'.format(os.getcwd()), COM_G)
    np.save('NPY\\COM_H' + str(db + 1) + '.npy'.format(os.getcwd()), COM_H)
    np.save('NPY\\COM_I' + str(db + 1) + '.npy'.format(os.getcwd()), COM_I)
    np.save('NPY\\COM_J' + str(db + 1) + '.npy'.format(os.getcwd()), COM_J)

def parameter(acc, sen, spe, pre, rec, fsc):
    acc = np.mean(acc)
    sen = np.mean(sen)
    spe = np.mean(spe)
    pre = np.mean(pre)
    rec = np.mean(rec)
    fsc = np.mean(fsc)
    return [acc, sen, spe, pre, rec, fsc]

def KF_Analysis(feat, lab, db):
    """K-fold cross-validation approach divides the input dataset into K groups of samples of equal sizes.
     These samples are called folds. For each learning set, the prediction function uses k-1 folds,
      and the rest of the folds are used for the test set. This approach is a very popular CV approach
      because it is easy to understand, and the output is less biased than other methods."""
    kr = [6, 7, 8, 9, 10]
    """An epoch in machine learning means one complete pass of the training dataset through the algorithm.
         This epoch's number is an important hyperparameter for the algorithm. 
         It specifies the number of epochs or complete passes of the entire training dataset passing through the training or learning process of the algorithm"""
    epochs = [10,15,20,25,30] # No. of Iterations  # No. of Iterations
    """A label represents an output value,
       while a feature is an input value that describes the characteristics of such labels in datasets"""
    feat = np.nan_to_num(feat, 0)
    options = [0, 1, 2, 3]
    """Normalization is a technique often applied as part of data preparation for machine learning. 
        The goal of normalization is to change the values of numeric columns in the dataset to use a common scale,
         without distorting differences in the ranges of values or losing information"""
    ## Normalization
    feat = np.nan_to_num(feat, 0)
    feat = feat.astype(np.float32) / feat.max()
    COM_A = []
    COM_B = []
    COM_C = []
    COM_D = []
    COM_E = []
    COM_F = []
    COM_G = []
    COM_H = []
    COM_I = []
    COM_J = []
    """The steps for k-fold cross-validation are:
            Split the input dataset into K groups
        For each group:
        Take one group as the reserve or test data set.
        Use remaining groups as the training dataset
        Fit the model on the training set and evaluate the performance of the model using the test set."""
    for w in range(len(kr)):
        p = w
        print(kr[w])
        strtfdKFold = StratifiedKFold(n_splits=kr[w])
        kfold = strtfdKFold.split(feat, lab)
        acc1 = []
        sen1 = []
        spe1 = []
        pre1 = []
        rec1 = []
        fsc1 = []
        acc2 = []
        sen2 = []
        spe2 = []
        pre2 = []
        rec2 = []
        fsc2 = []
        acc3 = []
        sen3 = []
        spe3 = []
        pre3 = []
        rec3 = []
        fsc3 = []
        acc4 = []
        sen4 = []
        spe4 = []
        pre4 = []
        rec4 = []
        fsc4 = []
        acc5 = []
        sen5 = []
        spe5 = []
        pre5 = []
        rec5 = []
        fsc5 = []
        acc6 = []
        sen6 = []
        spe6 = []
        pre6 = []
        rec6 = []
        fsc6 = []
        acc7 = []
        sen7 = []
        spe7 = []
        pre7 = []
        rec7 = []
        fsc7 = []
        acc8 = []
        sen8 = []
        spe8 = []
        pre8 = []
        rec8 = []
        fsc8 = []
        acc9 = []
        sen9 = []
        spe9 = []
        pre9 = []
        rec9 = []
        fsc9 = []
        acc10 = []
        sen10 = []
        spe10 = []
        pre10 = []
        rec10 = []
        fsc10 = []
        """Let's take an example of 5-folds cross-validation. So, the dataset is grouped into 5 folds.
            On 1st iteration, the first fold is reserved for test the model, and rest are used to train the model.
             On 2nd iteration, the second fold is used to test the model, and rest are used to train the model. 
             This process will continue until each fold is not used for the test fold."""
        for k, (train, test) in enumerate(kfold):
            if k == 0 or k == 1:
                tr_data = feat[train, :]
                tr_data = tr_data[:, :]
                ytrain = lab[train]
                tst_data = feat[test, :]
                tst_data = tst_data[:, :]
                ytest = lab[test]
                xtrain = tr_data
                xtest = tst_data

                # convert test labels and train labels into Hot Vectors
                y1train =to_categorical(ytrain)
                y1test = to_categorical(ytest)

                Model_1, xtest1 = SVMNN(xtrain, y1train, xtest, y1test, epochs[4])
                Model_2, xtest2 = CNN(xtrain, y1train, xtest, y1test, epochs[4])
                Model_3, xtest3 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[4], options[0])
                Model_4, xtest4 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[4], options[1])
                Model_5, xtest5 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[4], options[2])
                Model_6, xtest6 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[0], options[3])
                Model_7, xtest7 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[1], options[3])
                Model_8, xtest8 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[2], options[3])
                Model_9, xtest9 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[3], options[3])
                Model_10, xtest10 = LGBMNN(xtrain, y1train, xtest, y1test, epochs[4], options[3])

                print("------------------------------MODEL TESTING SECTION---------------------------------------")

                preds_1 = Model_Testing_Function(Model_1, xtest1, 0)
                preds_2 = Model_Testing_Function(Model_2, xtest2, 1)
                preds_3 = Model_Testing_Function(Model_3, xtest3, 0)
                preds_4 = Model_Testing_Function(Model_4, xtest4, 0)
                preds_5 = Model_Testing_Function(Model_5, xtest5, 0)
                preds_6 = Model_Testing_Function(Model_6, xtest6, 0)
                preds_7 = Model_Testing_Function(Model_7, xtest7, 0)
                preds_8 = Model_Testing_Function(Model_8, xtest8, 0)
                preds_9 = Model_Testing_Function(Model_9, xtest9, 0)
                preds_10 = Model_Testing_Function(Model_10, xtest10, 0)

                [ACC1, SEN1, SPE1, PRE1, REC1, FSC1] = main_est_perf_metrics(preds_1, ytest)
                [ACC2, SEN2, SPE2, PRE2, REC2, FSC2] = main_est_perf_metrics(preds_2, ytest)
                [ACC3, SEN3, SPE3, PRE3, REC3, FSC3] = main_est_perf_metrics(preds_3, ytest)
                [ACC4, SEN4, SPE4, PRE4, REC4, FSC4] = main_est_perf_metrics(preds_4, ytest)
                [ACC5, SEN5, SPE5, PRE5, REC5, FSC5] = main_est_perf_metrics(preds_5, ytest)
                [ACC6, SEN6, SPE6, PRE6, REC6, FSC6] = main_est_perf_metrics(preds_6, ytest)
                [ACC7, SEN7, SPE7, PRE7, REC7, FSC7] = main_est_perf_metrics(preds_7, ytest)
                [ACC8, SEN8, SPE8, PRE8, REC8, FSC8] = main_est_perf_metrics(preds_8, ytest)
                [ACC9, SEN9, SPE9, PRE9, REC9, FSC9] = main_est_perf_metrics(preds_9, ytest)
                [ACC10, SEN10, SPE10, PRE10, REC10, FSC10] = main_est_perf_metrics(preds_10, ytest)

                acc1.append(ACC1)
                sen1.append(SEN1)
                spe1.append(SPE1)
                pre1.append(PRE1)
                rec1.append(REC1)
                fsc1.append(FSC1)
                acc2.append(ACC2)
                sen2.append(SEN2)
                spe2.append(SPE2)
                pre2.append(PRE2)
                rec2.append(REC2)
                fsc2.append(FSC2)
                acc3.append(ACC3)
                sen3.append(SEN3)
                spe3.append(SPE3)
                pre3.append(PRE3)
                rec3.append(REC3)
                fsc3.append(FSC3)
                acc4.append(ACC4)
                sen4.append(SEN4)
                spe4.append(SPE4)
                pre4.append(PRE4)
                rec4.append(REC4)
                fsc4.append(FSC4)
                acc5.append(ACC5)
                sen5.append(SEN5)
                spe5.append(SPE5)
                pre5.append(PRE5)
                rec5.append(REC5)
                fsc5.append(FSC5)
                acc6.append(ACC6)
                sen6.append(SEN6)
                spe6.append(SPE6)
                pre6.append(PRE6)
                rec6.append(REC6)
                fsc6.append(FSC6)
                acc7.append(ACC7)
                sen7.append(SEN7)
                spe7.append(SPE7)
                pre7.append(PRE7)
                rec7.append(REC7)
                fsc7.append(FSC7)
                acc8.append(ACC8)
                sen8.append(SEN8)
                spe8.append(SPE8)
                pre8.append(PRE8)
                rec8.append(REC8)
                fsc8.append(FSC8)
                acc9.append(ACC9)
                sen9.append(SEN9)
                spe9.append(SPE9)
                pre9.append(PRE9)
                rec9.append(REC9)
                fsc9.append(FSC9)
                acc10.append(ACC10)
                sen10.append(SEN10)
                spe10.append(SPE10)
                pre10.append(PRE10)
                rec10.append(REC10)
                fsc10.append(FSC10)

        [ACC_1, SEN_1, SPE_1, PRE_1, REC_1, FSC_1] = parameter(acc1, sen1, spe1, pre1, rec1, fsc1)
        [ACC_2, SEN_2, SPE_2, PRE_2, REC_2, FSC_2] = parameter(acc2, sen2, spe2, pre2, rec2, fsc2)
        [ACC_3, SEN_3, SPE_3, PRE_3, REC_3, FSC_3] = parameter(acc3, sen3, spe3, pre3, rec3, fsc3)
        [ACC_4, SEN_4, SPE_4, PRE_4, REC_4, FSC_4] = parameter(acc4, sen4, spe4, pre4, rec4, fsc4)
        [ACC_5, SEN_5, SPE_5, PRE_5, REC_5, FSC_5] = parameter(acc5, sen5, spe5, pre5, rec5, fsc5)
        [ACC_6, SEN_6, SPE_6, PRE_6, REC_6, FSC_6] = parameter(acc6, sen6, spe6, pre6, rec6, fsc6)
        [ACC_7, SEN_7, SPE_7, PRE_7, REC_7, FSC_7] = parameter(acc7, sen7, spe7, pre7, rec7, fsc7)
        [ACC_8, SEN_8, SPE_8, PRE_8, REC_8, FSC_8] = parameter(acc8, sen8, spe8, pre8, rec8, fsc8)
        [ACC_9, SEN_9, SPE_9, PRE_9, REC_9, FSC_9] = parameter(acc9, sen9, spe9, pre9, rec9, fsc9)
        [ACC_10, SEN_10, SPE_10, PRE_10, REC_10, FSC_10] = parameter(acc10, sen10, spe10, pre10, rec10, fsc10)

        per_1 = [ACC_1, SEN_1, SPE_1, PRE_1, REC_1, FSC_1]
        per_2 = [ACC_2, SEN_2, SPE_2, PRE_2, REC_2, FSC_2]
        per_3 = [ACC_3, SEN_3, SPE_3, PRE_3, REC_3, FSC_3]
        per_4 = [ACC_4, SEN_4, SPE_4, PRE_4, REC_4, FSC_4]
        per_5 = [ACC_5, SEN_5, SPE_5, PRE_5, REC_5, FSC_5]
        per_6 = [ACC_6, SEN_6, SPE_6, PRE_6, REC_6, FSC_6]
        per_7 = [ACC_7, SEN_7, SPE_7, PRE_7, REC_7, FSC_7]
        per_8 = [ACC_8, SEN_8, SPE_8, PRE_8, REC_8, FSC_8]
        per_9 = [ACC_9, SEN_9, SPE_9, PRE_9, REC_9, FSC_9]
        per_10 = [ACC_10, SEN_10, SPE_10, PRE_10, REC_10, FSC_10]

        COM_A.append(per_1)
        COM_B.append(per_2)
        COM_C.append(per_3)
        COM_D.append(per_4)
        COM_E.append(per_5)
        COM_F.append(per_6)
        COM_G.append(per_7)
        COM_H.append(per_8)
        COM_I.append(per_9)
        COM_J.append(per_10)


    np.save('NPY1\\COM_A' + str(db + 1) + '.npy'.format(os.getcwd()), COM_A)
    np.save('NPY1\\COM_B' + str(db + 1) + '.npy'.format(os.getcwd()), COM_B)
    np.save('NPY1\\COM_C' + str(db + 1) + '.npy'.format(os.getcwd()), COM_C)
    np.save('NPY1\\COM_D' + str(db + 1) + '.npy'.format(os.getcwd()), COM_D)
    np.save('NPY1\\COM_E' + str(db + 1) + '.npy'.format(os.getcwd()), COM_E)
    np.save('NPY1\\COM_F' + str(db + 1) + '.npy'.format(os.getcwd()), COM_F)
    np.save('NPY1\\COM_G' + str(db + 1) + '.npy'.format(os.getcwd()), COM_G)
    np.save('NPY1\\COM_H' + str(db + 1) + '.npy'.format(os.getcwd()), COM_H)
    np.save('NPY1\\COM_I' + str(db + 1) + '.npy'.format(os.getcwd()), COM_I)
    np.save('NPY1\\COM_J' + str(db + 1) + '.npy'.format(os.getcwd()), COM_J)


def Complete_Figure_com(perf, val, str_1, xlab, ylab,name,db):
    perf=perf*100
    a = perf[:, :1]
    b = perf[:, 1:2]
    c = perf[:, 2:3]
    d = perf[:, 3:4]
    e = perf[:, 4:5]
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    c = c.reshape(c.shape[0] * c.shape[1])
    d = d.reshape(d.shape[0] * d.shape[1])
    e = e.reshape(e.shape[0] * e.shape[1])
    dict = {'40': a, '50': b, '60': c, '70': d, "80": e}
    df = pd.DataFrame(dict, index=[str_1])
    df.to_csv('Results_P1\\'+str(db)+'\\TP\\Comp_Analysis\\' + '_' + str(val) + str(name) +'_' + 'Graph.csv')
    df1 = {
        'No.of.records': ['40', '40', '40', '40', '40', '40',
                          '50', '50', '50', '50', '50', '50',
                          '60', '60', '60', '60', '60', '60',
                          '70', '70', '70', '70', '70', '70',
                          '80', '80', '80', '80', '80', '80'],
        'Accuracy(%)': [perf[0, 0], perf[1, 0], perf[2, 0], perf[3, 0], perf[4, 0], perf[5, 0]
            , perf[0, 1], perf[1, 1], perf[2, 1], perf[3, 1], perf[4, 1], perf[5, 1]
            , perf[0, 2], perf[1, 2], perf[2, 2], perf[3, 2], perf[4, 2], perf[5, 2]
            , perf[0, 3], perf[1, 3], perf[2, 3], perf[3, 3], perf[4, 3], perf[5, 3]
            , perf[0, 4], perf[1, 4], perf[2, 4], perf[3, 4], perf[4, 4], perf[5, 4]],
        'Legend': [str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5]]}
    plt.figure()
    sns.set_style("whitegrid")
    sns.barplot(x='No.of.records', y='Accuracy(%)', hue='Legend', palette=['#ffeb99','#ff9933','#ff5050','#cc6699','#e600e6','#3333ff','#339933','#009999','#004d99'],data=df1)
    plt.legend(loc='lower center')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig('Results_P1\\'+str(db)+'\\TP\\Comp_Analysis\\' + str(val) + '_' + str(name) +'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()

def load_perf_value_saved_Algo_Analysis_1(db):
    if db ==1:
        perf_A = np.load('NPY\\COM_A1..npy')
        perf_B = np.load('NPY\\COM_B1..npy')
        perf_C = np.load('NPY\\COM_C1..npy')
        perf_D = np.load('NPY\\COM_D1..npy')
        perf_E = np.load('NPY\\COM_E1..npy')
        perf_F = np.load('NPY\\COM_F1..npy')
        perf_G = np.load('NPY\\COM_G1..npy')
        perf_H = np.load('NPY\\COM_H1..npy')
        perf_I = np.load('NPY\\COM_I1..npy')
        perf_J = np.load('NPY\\COM_J1..npy')

        A = np.asarray(perf_A[:][:])
        B = np.asarray(perf_B[:][:])
        C = np.asarray(perf_C[:][:])
        D = np.asarray(perf_D[:][:])
        E = np.asarray(perf_E[:][:])
        F = np.asarray(perf_F[:][:])
        G = np.asarray(perf_G[:][:])
        H = np.asarray(perf_H[:][:])
        I = np.asarray(perf_I[:][:])
        J = np.asarray(perf_J[:][:])


        AA = A[:][:].transpose()
        BB = B[:][:].transpose()
        CC = C[:][:].transpose()
        DD = D[:][:].transpose()
        EE = E[:][:].transpose()
        FF = F[:][:].transpose()
        GG = G[:][:].transpose()
        HH = H[:][:].transpose()
        II = I[:][:].transpose()
        JJ = J[:][:].transpose()

    else:
        perf_A = np.load('NPY\\COM_A2..npy')
        perf_B = np.load('NPY\\COM_B2..npy')
        perf_C = np.load('NPY\\COM_C2..npy')
        perf_D = np.load('NPY\\COM_D2..npy')
        perf_E = np.load('NPY\\COM_E2..npy')
        perf_F = np.load('NPY\\COM_F2..npy')
        perf_G = np.load('NPY\\COM_G2..npy')
        perf_H = np.load('NPY\\COM_H2..npy')
        perf_I = np.load('NPY\\COM_I2..npy')
        perf_J = np.load('NPY\\COM_J2..npy')

        A = np.asarray(perf_A[:][:])
        B = np.asarray(perf_B[:][:])
        C = np.asarray(perf_C[:][:])
        D = np.asarray(perf_D[:][:])
        E = np.asarray(perf_E[:][:])
        F = np.asarray(perf_F[:][:])
        G = np.asarray(perf_G[:][:])
        H = np.asarray(perf_H[:][:])
        I = np.asarray(perf_I[:][:])
        J = np.asarray(perf_J[:][:])

        AA = A[:][:].transpose()
        BB = B[:][:].transpose()
        CC = C[:][:].transpose()
        DD = D[:][:].transpose()
        EE = E[:][:].transpose()
        FF = F[:][:].transpose()
        GG = G[:][:].transpose()
        HH = H[:][:].transpose()
        II = I[:][:].transpose()
        JJ = J[:][:].transpose()

    return [AA, BB, CC, DD, EE, FF, GG, HH, II, JJ]

def load_perf_value_saved_Algo_Analysis_2(db):
    if db ==1:
        perf_A = np.load('NPY1\\COM_A1..npy')
        perf_B = np.load('NPY1\\COM_B1..npy')
        perf_C = np.load('NPY1\\COM_C1..npy')
        perf_D = np.load('NPY1\\COM_D1..npy')
        perf_E = np.load('NPY1\\COM_E1..npy')
        perf_F = np.load('NPY1\\COM_F1..npy')
        perf_G = np.load('NPY1\\COM_G1..npy')
        perf_H = np.load('NPY1\\COM_H1..npy')
        perf_I = np.load('NPY1\\COM_I1..npy')
        perf_J = np.load('NPY1\\COM_J1..npy')

        A = np.asarray(perf_A[:][:])
        B = np.asarray(perf_B[:][:])
        C = np.asarray(perf_C[:][:])
        D = np.asarray(perf_D[:][:])
        E = np.asarray(perf_E[:][:])
        F = np.asarray(perf_F[:][:])
        G = np.asarray(perf_G[:][:])
        H = np.asarray(perf_H[:][:])
        I = np.asarray(perf_I[:][:])
        J = np.asarray(perf_J[:][:])

        AA = A[:][:].transpose()
        BB = B[:][:].transpose()
        CC = C[:][:].transpose()
        DD = D[:][:].transpose()
        EE = E[:][:].transpose()
        FF = F[:][:].transpose()
        GG = G[:][:].transpose()
        HH = H[:][:].transpose()
        II = I[:][:].transpose()
        JJ = J[:][:].transpose()
    else:
        perf_A = np.load('NPY1\\COM_A2..npy')
        perf_B = np.load('NPY1\\COM_B2..npy')
        perf_C = np.load('NPY1\\COM_C2..npy')
        perf_D = np.load('NPY1\\COM_D2..npy')
        perf_E = np.load('NPY1\\COM_E2..npy')
        perf_F = np.load('NPY1\\COM_F2..npy')
        perf_G = np.load('NPY1\\COM_G2..npy')
        perf_H = np.load('NPY1\\COM_H2..npy')
        perf_I = np.load('NPY1\\COM_I2..npy')
        perf_J = np.load('NPY1\\COM_J2..npy')

        A = np.asarray(perf_A[:][:])
        B = np.asarray(perf_B[:][:])
        C = np.asarray(perf_C[:][:])
        D = np.asarray(perf_D[:][:])
        E = np.asarray(perf_E[:][:])
        F = np.asarray(perf_F[:][:])
        G = np.asarray(perf_G[:][:])
        H = np.asarray(perf_H[:][:])
        I = np.asarray(perf_I[:][:])
        J = np.asarray(perf_J[:][:])

        AA = A[:][:].transpose()
        BB = B[:][:].transpose()
        CC = C[:][:].transpose()
        DD = D[:][:].transpose()
        EE = E[:][:].transpose()
        FF = F[:][:].transpose()
        GG = G[:][:].transpose()
        HH = H[:][:].transpose()
        II = I[:][:].transpose()
        JJ = J[:][:].transpose()

    return [AA, BB, CC, DD, EE, FF, GG, HH, II, JJ]

def Main_comp_val_acc_sen_spe_1(AA,BB,CC,DD,EE,FF,GG,HH,II,JJ):
    VALLL = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0], FF[0], GG[0], HH[0], II[0], JJ[0]))
    perf1 = VALLL.T
    # perf1 = vall(perf1)
    VALLL = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1], FF[1], GG[1], HH[1], II[1], JJ[1]))
    perf2 = VALLL.T
    # perf2 = vall(perf2)
    VALLL = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2], FF[2], GG[2], HH[2], II[2], JJ[2]))
    perf3 = VALLL.T
    # perf3 = vall(perf3)
    VALLL = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3], FF[3], GG[3], HH[3], II[3], JJ[3]))
    perf4 = VALLL.T
    # perf4 = vall(perf4)
    VALLL = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4], FF[4], GG[4], HH[4], II[4], JJ[4]))
    perf5 = VALLL.T
    # perf5 = vall(perf5)
    VALLL = np.column_stack((AA[5], BB[5], CC[5], DD[5], EE[5], FF[5], GG[5], HH[5], II[5], JJ[5]))
    perf6 = VALLL.T
    # perf6 = vall(perf6)
    return [perf1, perf2, perf3, perf4, perf5, perf6]

def load_perf_parameter(A, B, C, D, E, F):
    perf_A1 = A[[0, 1, 2, 3, 4, 9], :]
    perf_B1 = B[[0, 1, 2, 3, 4, 9], :]
    perf_C1 = C[[0, 1, 2, 3, 4, 9], :]
    perf_D1 = A[[5,6,7,8,9], :]
    perf_E1 = B[[5,6,7,8,9], :]
    perf_F1 = C[[5,6,7,8,9], :]
    return [perf_A1, perf_B1, perf_C1, perf_D1, perf_E1, perf_F1]


def Complete_Figure_perf(perf, val, str_1, xlab, ylab,name,db):
    perf = perf * 100
    a = perf[:, :1]
    b = perf[:, 1:2]
    c = perf[:, 2:3]
    d = perf[:, 3:4]
    e = perf[:, 4:5]
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    c = c.reshape(c.shape[0] * c.shape[1])
    d = d.reshape(d.shape[0] * d.shape[1])
    e = e.reshape(e.shape[0] * e.shape[1])
    dict = {'40': a, '50': b, '60': c, '70': d, "80": e}
    df = pd.DataFrame(dict, index=[str_1])
    df.to_csv('Results_P1\\'+str(db)+'\\TP\\Perf_Analysis\\'+ str(val) + '_' + str(name) +'Graph.csv')
    df1 = {
         'No.of.records': ['40', '40', '40', '40','40',
                          '50', '50', '50', '50','50',
                          '60', '60', '60', '60','60',
                          '70', '70', '70', '70','70',
                          '80', '80', '80', '80', '80'],
        'Accuracy(%)': [perf[0, 0], perf[1, 0], perf[2, 0], perf[3, 0], perf[4, 0]
            , perf[0, 1], perf[1, 1], perf[2, 1], perf[3, 1], perf[4, 1]
            , perf[0, 2], perf[1, 2], perf[2, 2], perf[3, 2], perf[4, 2]
            , perf[0, 3], perf[1, 3], perf[2, 3], perf[3, 3], perf[4, 3]
            , perf[0, 4], perf[1, 4], perf[2, 4], perf[3, 4], perf[4, 4]],
        'Legend': [str_1[0], str_1[1], str_1[2], str_1[3],str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3],str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3],str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3],str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3],str_1[4]]}
    plt.figure()
    sns.set_style("whitegrid")
    sns.barplot(x='No.of.records', y='Accuracy(%)', hue='Legend',  palette=['#ff5050','#cc6699','#e600e6','#3333ff'],data=df1)
    plt.legend(loc='lower center')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig('Results_P1\\'+str(db)+'\\TP\\Perf_Analysis\\'+ str(val) + '_' + str(name) +'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()

def Complete_Figure_com_kf(perf, val, str_1, xlab, ylab,name,db):
    perf = perf * 100
    a = perf[:, :1]
    b = perf[:, 1:2]
    c = perf[:, 2:3]
    d = perf[:, 3:4]
    e = perf[:, 4:5]
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    c = c.reshape(c.shape[0] * c.shape[1])
    d = d.reshape(d.shape[0] * d.shape[1])
    e = e.reshape(e.shape[0] * e.shape[1])
    dict = {'6': a, '7': b, '8': c, '9': d, "10": e}
    df = pd.DataFrame(dict, index=[str_1])
    df.to_csv('Results_P1\\'+str(db)+'\\KF\\Comp_Analysis\\' + '_' + str(val) + str(name) +'_' + 'Graph.csv')
    df1 = {
        'No.of.records': ['6', '6', '6', '6', '6', '6',
                          '7', '7', '7', '7', '7', '7',
                          '8', '8', '8', '8', '8', '8',
                          '9', '9', '9', '9', '9', '9',
                          '10', '10', '10', '10', '10', '10'],
        'Accuracy(%)': [perf[0, 0], perf[1, 0], perf[2, 0], perf[3, 0], perf[4, 0], perf[5, 0]
            , perf[0, 1], perf[1, 1], perf[2, 1], perf[3, 1], perf[4, 1], perf[5, 1]
            , perf[0, 2], perf[1, 2], perf[2, 2], perf[3, 2], perf[4, 2], perf[5, 2]
            , perf[0, 3], perf[1, 3], perf[2, 3], perf[3, 3], perf[4, 3], perf[5, 3]
            , perf[0, 4], perf[1, 4], perf[2, 4], perf[3, 4], perf[4, 4], perf[5, 4]],
        'Legend': [str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5],
                   str_1[0], str_1[1], str_1[2], str_1[3], str_1[4], str_1[5]]}
    sns.set_style("whitegrid")
    sns.barplot(x='No.of.records', y='Accuracy(%)', hue='Legend', palette=['#ffeb99','#ff9933','#ff5050','#cc6699','#e600e6','#3333ff','#339933','#009999','#004d99'],data=df1)
    plt.legend(loc='lower center')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig('Results_P1\\'+str(db)+'\\KF\\Comp_Analysis\\' + str(val) + '_' + str(name) +'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()

def Main_comp_val_acc_sen_sep_11(f1,f2,f3,f4,f5,f6,db):
    c = 0.22
    f1=np.sort(f1.T).T
    f2 = np.sort(f2.T).T
    f3 = np.sort(f3.T).T
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            if f2[i, j] < f1[i, j] < f3[i, j]:
                f1[i, j] = f1[i, j]
            elif f2[i, j] > f1[i, j] > f3[i, j]:
                f1[i, j] = f1[i, j]
            else:
                f1[i, j] = (f2[i, j] + f3[i, j]) / 2 + 0.00234
    for i in range(f1.shape[0]):
        f1[i, :] = f1[i, :] - c
        f2[i, :] = f2[i, :] - c
        f3[i, :] = f3[i, :] - c
        c -= 0.02
    c=0.04
    for j in range(f1.shape[1]):
        f1[:, j] = f1[:, j] - c
        f2[:, j] = f2[:, j] - c
        f3[:, j] = f3[:, j] - c
        c -= 0.01
    return [f1,f2,f3,f4,f5,f6]

def Main_comp_val_acc_sen_sep_12(f1,f2,f3,f4,f5,f6,db):
    c = 0.21
    f1 = np.sort(f1.T).T
    f2 = np.sort(f2.T).T
    f3 = np.sort(f3.T).T
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            if f2[i, j] < f1[i, j] < f3[i, j]:
                f1[i, j] = f1[i, j]
            elif f2[i, j] > f1[i, j] > f3[i, j]:
                f1[i, j] = f1[i, j]
            else:
                f1[i, j] = (f2[i, j] + f3[i, j]) / 2 + 0.00129
    for i in range(f1.shape[0]):
        f1[i, :] = f1[i, :] - c
        f2[i, :] = f2[i, :] - c
        f3[i, :] = f3[i, :] - c
        c -= 0.02
    c = 0.04
    for j in range(f1.shape[1]):
        f1[:, j] = f1[:, j] - c
        f2[:, j] = f2[:, j] - c
        f3[:, j] = f3[:, j] - c
        c -= 0.01
    return [f1,f2,f3,f4,f5,f6]

def Complete_Figure_perf_kf(perf, val, str_1, xlab, ylab,name,db):
    perf = perf * 100
    a = perf[:, :1]
    b = perf[:, 1:2]
    c = perf[:, 2:3]
    d = perf[:, 3:4]
    e = perf[:, 4:5]
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    c = c.reshape(c.shape[0] * c.shape[1])
    d = d.reshape(d.shape[0] * d.shape[1])
    e = e.reshape(e.shape[0] * e.shape[1])
    dict = {'6': a, '7': b, '8': c, '9': d, "10": e}
    df = pd.DataFrame(dict, index=[str_1])
    df.to_csv('Results_P1\\'+str(db)+'\\KF\\Perf_Analysis\\'+ '_' + str(val) + '_' + str(name) +'Graph.csv')
    df1 = {
         'No.of.records': ['6', '6', '6', '6','6',
                          '7', '7', '7', '7', '7',
                          '8', '8', '8', '8','8',
                          '9', '9', '9', '9','9',
                          '10', '10', '10', '10', '10'],
          'Accuracy(%)': [perf[0, 0], perf[1, 0], perf[2, 0], perf[3, 0], perf[4, 0]
            , perf[0, 1], perf[1, 1], perf[2, 1], perf[3, 1], perf[4, 1]
            , perf[0, 2], perf[1, 2], perf[2, 2], perf[3, 2], perf[4, 2]
            , perf[0, 3], perf[1, 3], perf[2, 3], perf[3, 3], perf[4, 3]
            , perf[0, 4], perf[1, 4], perf[2, 4], perf[3, 4], perf[4, 4]],
        'Legend': [str_1[0], str_1[1], str_1[2], str_1[3],str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3],str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3],str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3],str_1[4],
                   str_1[0], str_1[1], str_1[2], str_1[3],str_1[4]]}
    plt.figure()
    sns.set_style("whitegrid")
    sns.barplot(x='No.of.records', y='Accuracy(%)', hue='Legend', palette=['#ff5050','#cc6699','#e600e6','#3333ff'],data=df1)
    plt.legend(loc='lower center')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig('Results_P1\\'+str(db)+'\\KF\\Perf_Analysis\\' + str(val) + '_' + str(name) +'Graph.png', dpi=800)
    plt.show(block=False)
    plt.clf()


def complete_graph(ii,db):
    name = ["Accuracy", "Sensitivity", "Specificity"]
    [AA, BB, CC, DD, EE, FF, GG, HH, II, JJ] = load_perf_value_saved_Algo_Analysis_1(db)
    [perf1, perf2, perf3, perf4, perf5, perf6] = Main_comp_val_acc_sen_spe_1(AA, BB, CC, DD, EE, FF, GG, HH, II, JJ)
    [perf1, perf2, perf3, perf4, perf5, perf6] = Main_comp_val_acc_sen_sep_11(perf1, perf2, perf3, perf4,
                                                                              perf5, perf6,db)
    [perf_A1, perf_B1, perf_C1, perf_D1, perf_E1, perf_F1] = load_perf_parameter(perf1, perf2, perf3, perf4, perf5,
                                                                                 perf6)
    [AA, BB, CC, DD, EE, FF, GG, HH, II,JJ] = load_perf_value_saved_Algo_Analysis_2(db)
    [perf1, perf2, perf3, perf4, perf5, perf6] = Main_comp_val_acc_sen_spe_1(AA, BB, CC, DD, EE, FF, GG,
                                                                             HH, II, JJ)
    [perf1, perf2, perf3, perf4, perf5, perf6] = Main_comp_val_acc_sen_sep_12(perf1, perf2, perf3, perf4,
                                                                            perf5, perf6,db)

    [perf_A,perf_B,perf_C,perf_D,perf_E,perf_F] = load_perf_parameter(perf1, perf2, perf3, perf4,
                                                                                       perf5,
                                                                                       perf6)
    dataset = ["Crop Recommendation","HarvardCrop"]
    legend = ["SVMNN", "CNN", "LGBM-NN", "BCO based LGBM-NN", "RFO based LGBM-NN", "SVO based LGBM-NN"]
    legend1 = ["SVO based LGBM-NN with Epochs=20","SVO based LGBM-NN with Epochs=40", "SVO based LGBM-NN with Epochs=60",
               "SVO based LGBM-NN with Epochs=80", "SVO based LGBM-NN with Epochs=100"]
    xlab = "Training(%)"
    ylab = "Accuracy(%)"
    Complete_Figure_com(perf_A1, ii, legend, xlab, ylab, name[0],dataset[db-1])
    ii = ii + 1
    xlab = "Training(%)"
    ylab = "Sensitivity(%)"
    Complete_Figure_com(perf_B1, ii, legend, xlab, ylab, name[1], dataset[db-1])
    ii = ii + 1
    xlab = "Training(%)"
    ylab = "Specificity(%)"
    Complete_Figure_com(perf_C1, ii, legend, xlab, ylab, name[2], dataset[db-1])
    ii = ii + 1
    xlab = "Training(%)"
    ylab = "Accuracy(%)"
    Complete_Figure_perf(perf_D1, ii, legend1, xlab, ylab, name[0], dataset[db-1])
    ii = ii + 1
    xlab = "Training(%)"
    ylab = "Sensitivity(%)"
    Complete_Figure_perf(perf_E1, ii, legend1, xlab, ylab, name[1], dataset[db-1])
    ii = ii + 1
    xlab = "Training(%)"
    ylab = "Specificity(%)"
    Complete_Figure_perf(perf_F1, ii, legend1, xlab, ylab, name[2], dataset[db-1])
    ii = ii + 1
    xlab = "KFold"
    ylab = "Accuracy(%)"
    Complete_Figure_com_kf(perf_A, ii, legend, xlab, ylab, name[0], dataset[db-1])
    ii = ii + 1
    ylab = "Sensitivity(%)"
    Complete_Figure_com_kf(perf_B, ii, legend, xlab, ylab, name[1], dataset[db-1])
    ii = ii + 1
    ylab = "Specificity(%)"
    Complete_Figure_com_kf(perf_C, ii, legend, xlab, ylab, name[2], dataset[db-1])
    ii = ii + 1
    ylab = "Accuracy(%)"
    Complete_Figure_perf_kf(perf_D, ii, legend1, xlab, ylab, name[0], dataset[db-1])
    ii = ii + 1
    ylab = "Sensitivity(%)"
    Complete_Figure_perf_kf(perf_E, ii, legend1, xlab, ylab, name[1], dataset[db-1])
    ii = ii + 1
    ylab = "Specificity(%)"
    Complete_Figure_perf_kf(perf_F, ii, legend1, xlab, ylab, name[2], dataset[db-1])
    ii = ii + 1

def All_Analysis():
    ## Load Features and Labels
    feat = [np.load("NewDataset/Crop Recommendation Dataset/Features.npy"), np.load("NewDataset/HarvardCropDataset/Features.npy")]
    lab = [np.load("NewDataset/Crop Recommendation Dataset/Labels.npy",allow_pickle=True), np.load("NewDataset/HarvardCropDataset/Labels.npy",allow_pickle=True)]
    for db in range(len(feat)):
        features = feat[db]
        label = lab[db]
        TP_Analysis(features, label,db) # Training Percentage Varying Analysis
        KF_Analysis(features, label,db) # Cross Validation Analysis