#Classification reports and confusion matrices are great methods to quantitatively evaluate model performance, while ROC curves provide a way to visually evaluate models. 
# https://en.wikipedia.org/wiki/Precision_and_recall#Imbalanced_data: According to Saito and Rehmsmeier, precision-recall plots are more informative than ROC plots when evaluating binary classifiers on imbalanced data. In such scenarios, ROC plots may be visually deceptive with respect to conclusions about the reliability of classification performance
# can calculate a number of metrics directly from cm
    # Accuracy is the fraction of correct predictions
    # precision: true positives / true positives +false positives
        # high precision indicates a low false positive rate
        # proportion of positive results that were correctly classified 
        # imbalanced dataset with lots of negative class vs positive --> precision more useful to fpr, because precision doesn't include true negatives and is not affected by imbalance (i.e. you will have more datapoints at the bottom of the curve not meeting the threshold when you have an imbalancced dataset with number of negative class > positive class)
    # recall: true positives / true positives + false negatives
        # high recall means most positives correctly 
        # recall is better than fpr as it doesn't rely on false negatives 
    # F1score: (2 * precision * recall) / (precision + recall)

# https://www.jeremyjordan.me/evaluating-a-machine-learning-model/
# As I discussed previously, it's important to use new data when evaluating our model to prevent the likelihood of overfitting to the training set. However, sometimes it's useful to evaluate our model as we're building it to find that best parameters of a model - but we can't use the test set for this evaluation or else we'll end up selecting the parameters that perform best on the test data but maybe not the parameters that generalize best. To evaluate the model while still building and tuning the model, we create a third subset of the data known as the validation set. A typical train/test/validation split would be to use 60% of the data for training, 20% of the data for validation, and 20% of the data for testing.
# I'll also note that it's very important to shuffle the data before making these splits so that each split has an accurate representation of the dataset.
#  precision vs recall: https://www.jeremyjordan.me/content/images/2017/07/Precisionrecall.svg.png
# Precision and recall are useful in cases where classes aren't evenly distributed. The common example is for developing a classification algorithm that predicts whether or not someone has a disease. If only a small percentage of the population (let's say 1%) has this disease, we could build a classifier that always predicts that the person does not have the disease, we would have built a model which is 99% accurate and 0% useful.
# However, if we measured the recall of this useless predictor, it would be clear that there was something wrong with our model. In this example, recall ensures that we're not overlooking the people who have the disease, while precision ensures that we're not misclassifying too many people as having the disease when they don't. Obviously, you wouldn't want a model that incorrectly predicts a person has cancer (the person would end up in a painful and expensive treatment process for a disease they didn't have) but you also don't want to incorrectly predict a person does not have cancer when in fact they do. Thus, it's important to evaluate both the precision and recall of a model.

# https://www.youtube.com/watch?v=wGw6R8AbcuI
# imbalanced datasets with skewed classes (in our case non-recurrence classes have lots more observations than recurrence) means that a non-learning algorithm predicting non-recurrence most of the time would only be wrong (85/286)~30% of the time 
# with skewed classes, can get high classification accuracy by just predicting more common (usualy negative) class more often --> accuracy not great evaluation metric --> need better evaluation metric 
# precision: of all the predicted positive class examples in the dataset, what proportion did we classify correctly
    # precision = true positives / predicted positives = true positives / true positive + false positive 
# recall: of all the actual positive class examples in the dataset, what proportion did we classify correclty 
    # recall = true positives / acutal positives = true positives / true positives + false negatives 

#https://www.youtube.com/watch?v=W5meQnGACGo
    #trade-off between precision and recall
        # very important to correctly classify recurrence events in order to prioritize treatment for patients (dont treat patients with cancer --> potential death) --> lower threshold, even at the risk of greater false positives --> lower precision, higher recall 
        # conversly, only want to subject people to painful treatments if we are very sure they have/could have cancer --> raise threshold --> higher precision, lower recall
        # It’s far worse if a patient with cancer is diagnosed as cancer-free, as opposed to a cancer-free patient being diagnosed with cancer only to realize later with more testing that he/she doesn't have it.
    # evaluating algorithms based on precision and recall means you use two evaluation metrics; difficult to choose optimal combination of two metrics which slows down decision-making --> A single evaluation metric is preferred. 
        # using geometric average of precision and recall [(precision + recall)/2] is not great because if you are setting a low threshold, you may have high recall (e.g. classifier predicts y=1 all the time) and consequently a relatively high average (and vice versa) --> but predicting only one class is not a very useful classifier when trying to generalize --> F-score
        # harmonic average is better as it accounts for extreme values in precision or recall. [2PR / (P+R)]. 
            # If Recall/precision is near 0, then numerator is zero and F score will be near zero. 
            # if recall/precision is near 1, then numerator is small and F score will be near zero
            # only get a high F score if both precision and recall and relatively large (and balanced values)
                # e.g. if P=1 and R=1. F-score = 1

    # reasonable approach: try a bunch of thresholds on cross-validation sets --> pick whichever value of threshold gives you the highest value on cross-validation set

# datacamp:
# Exploratory data analysis should always be the precursor to model building.
# hyperparameters are parameters that cannot be explicitly learnt by fitting model. choosing the correct values of hyperparameters makes or breaks a succesful model.
    # basic approach invovles trying a bunch of diff hyperparameter values, fit them seperately, evaluate and choose the best one
        # this is the current approach, figure out something better and it will make you famous 
    # essential to use k-fold cross validation as using train_test_split alone would risk overfitting hyperparameter to test set 
        # using all data for cross-validation may mean estimating model performance on any of it may not provide accurate picture on unseen data --> # split data into training and hold-out set at beginning --> perform cross-validation on training set and choose best hyperparameters --> evaluate model performance on hold-out set (which has not been used at all) to test how well model generalizes to unseen data 
    # Hyperparameters can affect each other!
# standardization
    # sutract each observation by the mean and divide by variance to standardize --> i.e. all features are centered around zero and have variance equal to one
    # subtract by minimum and divide by range --> min=0 and max=1
    # normalize so data ranges from -1 to +1
# model complexity
    # overfitting: model too complex --> low test accuracy
    # underfitting: model too simple --> low train accuracy 
    # too much regularization (small C) doesn't work well - due to underfitting - and too little regularization (large C) doesn't work well either - due to overfitting.
    # smaller values of C lead to less confident predictions. That's because smaller C means more regularization, which in turn means smaller coefficients, which means raw model outputs closer to zero and, thus, probabilities closer to 0.5 after the raw model output is squashed through the sigmoid function. 
# SVM
    # support vectors are incorrectly classified examples or correctly classified examples close to the decision boundary
    # how close datapoint have to be to count as support vectors controlled by regularisation strength
    # mathematically equivalent to hinge loss with L2 regularization
    # Kernel trick: fitting a linear model in a higher-dimensional space corresponds to fitting a non-linear model in the original feature space
    # Even though cross-validation already splits the training set into parts, it's often a good idea to hold out a separate test set to make sure the cross-validation results are sensible.
    # key hyperparameters: C, kernel, gamma (only with RBF)


# ROC and AUC: 
# https://www.youtube.com/watch?v=4jRBRDbJemM 
    # y-axis represents tpr (aka sensitivity); what proportion of positive class was correctly predicted
    # x-axis represnts fpr (1-specificity); what proportion of the negative class was incorrectly predicted
    # a binary classifier that is just randomly making guesses would be correct approximately 50% of the time --> resulting ROC curve would be a diagonal line in which the True Positive Rate and False Positive Rate are always equal. The Area under this ROC curve would be 0.5 --> AUC >0.5 means model is better than random 
    # Once you've plotted a ROC, you can choose the optimal threshold given the number of false positives you are willing to accept
    # AUC can help with model selection; greater the AUC, better the model
        # Imagine a ROC with only one point in top left corner (i.e. tpr=1, fpr=0). This is a great model --> greater area under roc means better model


# you want your split to reflect labels on data
# distribute labels in train and test sets as they are in original dataset
# can use stratify kwarg in sklearn train_test_split function, we need something similar to this 
# model complexity curve to show overfittings vs underfitting 

# https://www.youtube.com/watch?v=fSytzGwwBVw
# great vid for explaining cross-validation



# Cross-validation is a vital step in evaluating a model. It maximizes the amount of data that is used to train the model, as during the course of training, the model is not only trained, but also tested on all of the available data.
# Cross validation is essential but do not forget that the more folds you use, the more computationally expensive cross-validation becomes. 
# with k-fold CV:
    # from sklearn.model_selection import cross_val_score
    # cv_results = cross_val_score(reg, X, y, cv=5)
    # # print(cv_results)
    # np.mean(cv_results)

# import libraries

import cross_validation as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# import modules 
import process_visualise_data as pvd
import metrics

# import required Sklearn 
from sklearn.svm import LinearSVC
from sklearn import metrics as m

if __name__ == "__main__":

    # Show all df columns
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    # get data 
    original_df, df_onehot, headers, new_headers, targets, inputs = pvd.load_and_process_data("breast-cancer.data")
    
    
    # Basic exploratory analysis of working dataset 
    # print("Number of features: ", df_onehot.shape[1]-1)
    # print("Number of samples: ", df_onehot.shape[0], "\n")
    # print("info: \n", df_onehot.info())
    # print("Description: \n", df_onehot.describe())
    # print("Head of table \n", df_onehot.head())
    # print(df_onehot.shape)

    # visualise correlation class column
    # df_onehot.corr()['class'][1:].sort_values().plot(kind ='barh')
    # plt.title('Corrrelation between features and class')
    # plt.tight_layout()
    # plt.show()

    




    # create dataframe of feature columns
    features_df = pd.DataFrame(inputs)

    # turn targets into Panda Series
    response = pd.Series(targets) 
    
    # prepare data split with 5-fold validation 
    cross_validated_data = cv.split_k_fold(response, features_df, k=5)

    # iterate through each fold in the dataset 
    for fold in cross_validated_data: 

        # assign training and test splits 
        x_train = fold[0]
        x_test = fold[1]
        y_train = fold[2]
        y_test = fold[3]

        # create a linear SVM classifier
        clf = LinearSVC(random_state=0, max_iter=2000, dual=False)
        # clf_weak_regularisation = LinearSVC(random_state=0, max_iter=2000, dual=False, C=100)
        # clf_strong_regularisation = LinearSVC(random_state=0, max_iter=2000, dual=False, C=0.1)

        # train classifier using training set 
        clf.fit(x_train, y_train)
        # clf_weak_regularisation.fit(x_train, y_train)
        # clf_strong_regularisation.fit(x_train, y_train)

        # predict response variable for test set 
        y_pred = clf.predict(x_test)
        # y_pred_weak = clf_weak_regularisation.predict(x_test)
        # y_pred_strong = clf_weak_regularisation.predict(x_test)

        # training and validation accuracy.
        # print("Training accuracy: ", clf.score(x_train, y_train))
        # print("Weak Training accuracy: ", clf_weak_regularisation.score(x_train, y_train))
        # print("Strong Training accuracy: ", clf_weak_regularisation.score(x_train, y_train))
        # print("Test accuracy: ", clf.score(x_test, y_test))


        # print(metrics.accuracy(y_pred, y_test))
        # print("mine: ", metrics.confusion_matrix(y_pred, y_test))

# precision
            # from sklearn.metrics import precision_score, confusion_matrix
            # print("sklearn cm: ", confusion_matrix(y_test, y_pred))
            # print("sklearn: ", precision_score(y_test, y_pred))

            # cm = metrics.confusion_matrix(y_test, y_pred)
            # print("mine cm: ",cm)
            # print("mine: ", metrics.precision(cm))


# recall
        # from sklearn.metrics import recall_score, confusion_matrix
        # print("sklearn cm: ", confusion_matrix(y_test, y_pred))
        # print("sklearn Recall:",recall_score(y_test, y_pred))
        # print("\n")

        # cm = metrics.confusion_matrix(y_test, y_pred)
        # print("my cm: ",cm)
        # print("my Recall: ", metrics.recall(cm))
        # print("\n\n\n")

#f1 score
        # from sklearn.metrics import f1_score
        # print("sklearn f1:",f1_score(y_test, y_pred))
        
        # cm = metrics.confusion_matrix(y_test, y_pred)
        # precision = metrics.precision(cm)
        # recall = metrics.recall(cm)
        # print("my f1:",metrics.f1(precision, recall))



        # identifying support vectors
        # print("Number of original examples in training set", len(x_train))
        # print("Number of support vectors", len(clf.support_))
        # X_small = x_train[svm.support_]
        # y_small = y_train[svm.support_]


    #     # TODO: calculate metrics manually 
    #     # accuracy: how often is the classifier correct?
    #     # precision: what percentage of positive tuples are labeled as such?
    #     # recall: what percentage of positive tuples are labelled as such?
    #     # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #     # print("Precision:",metrics.precision_score(y_test, y_pred))
    #     # print("Recall:",metrics.recall_score(y_test, y_pred))
    #     # print("F1 score:",metrics.f1_score(y_test, y_pred))
    #     # cm = metrics.confusion_matrix(y_test, y_pred); 
    #     # # print("confusion matrix: ", cm)
    #     print(metrics.classification_report(y_test, y_pred))


        # from sklearn.model_selection import GridSearchCV
        # param_grid = {'C':[0.1,1,10,100]}
        # grid = GridSearchCV(clf,param_grid, )
        # grid.fit(x_train, y_train)
        # print(grid.best_params_)
        # print(grid.best_score_)
        # grid_predictions = grid.predict(x_test)
        # cm = metrics.confusion_matrix(y_test,grid_predictions)
        # print("confusion matrix: \n", cm)
        # print(metrics.classification_report(y_test,grid_predictions))

    #     # pred_grid = grid_svm.predict(x_test)
    #     # print(metrics.classification_report(y_test, pred_grid))





    #     # cm = np.array(metrics.confusion_matrix(y_test, y_pred, labels=[1,0]))
    #     # df_cm = pd.DataFrame(cm, index=['Recurrence', 'no-recurrence'], columns=['predicted recurrence', 'predicted non-recurrence'])
    #     # print(df_cm)
    #     # sns.heatmap(df_cm, annot=True)
    #     # plt.show


    #     # df_cm = pd.DataFrame(cm)
    #     # # print("confusion matrix: ", cm)
    #     # # print("DF confusion matrix: ", df_cm)
    #     # sns.heatmap(df_cm, annot=True)
    #     # # plt.subplots(figsize=(20,15))
    #     # plt.show

        
    # # run SVM and predict 




    # # split dataset into training and test data 
    # # training_inputs, training_targets, test_inputs, test_targets = pvd.split_train_test(inputs,targets,6151312,0.80)

    # # print(df_onehot.drop(columns="class"))


    # # features dataframe with headers
    # # features_df = pd.DataFrame(inputs, columns=new_headers[1:])

    # # print("training inputs: \n", training_inputs)
    # # print("training targets: \n", training_targets)

    # # print("test inputs: \n", test_inputs)
    # # print("test targets: \n", test_targets)



    # # view summary of features dataframe
    # # print(features_df.head())

    # # test
    # # sns.pairplot(original_df, hue = "class")
    # # plt.figure(figsize=(7,7))
    # # sns.heatmap(original_df['age menopause tumour_size inv-nodes node-caps deg-malig breast breast-quad irradiat class'.split()].corr(), annot=True)
    # # sns.scatterplot(x = 'tumour-size', y = 'deg-malig', hue = 'class', data = original_df)


    # # print("Original data:\n", original_df.head())




# def svm_hyperplane_plot(X_train, y_train, clf):

#         # fit model to training dataset
#         clf.fit(X_train, y_train)

#         # assign decision function
#         decision_function = np.dot(X_train, clf.coef_[0]) + clf.intercept_[0]

#         # support vectors are datapoints within margin boundaries 
#         support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
#         support_vectors = X_train[support_vector_indices]

#         fig = plt.figure(figsize=(10,10), dpi=80)
#         ax = fig.add_subplot(1,1,1)
#         plt.scatter(X_train[:, 3], X_train[:, 4], c=y_train, s=30, cmap=plt.cm.Paired)
#         plt.show

#     # for i, C in enumerate([1,100]):

#     #     # fit model to training dataset
#     #     clf.fit(X_train, y_train)

#     #     print(X_train)

#     #     # assign decision function
#     #     decision_function = np.dot(X_train, clf.coef_[0]) + clf.intercept_[0]

#     #     # support vectors are datapoints within margin boundaries 
#     #     support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
#     #     support_vectors = X_train[support_vector_indices]

#     #     # print("\n",support_vectors,"\n")

#     #     # prepare plot
#     #     plt.subplot(1, 2, i + 1)

#     #     # plot datapoints from two features in training set 
#     #     plt.scatter(X_train[:, 3], X_train[:, 4], c=y_train, s=30, cmap=plt.cm.Paired)

#     #     # get axes and limits
#     #     ax = plt.gca()
#     #     xlim = ax.get_xlim()
#     #     ylim = ax.get_ylim()

#     #     # create a mesh of x values and y vales 
#     #     xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
#     #                         np.linspace(ylim[0], ylim[1], 50))

#     #     # first flatten (by using ravel) and then combine xx and yy into a single 2d array
#     #     xx_yy = np.c_[xx.ravel(), yy.ravel()]

#     #     # print("ravel x: ", np.c_[xx.ravel(), yy.ravel()])
#     #     # print("ravel yy: ", yy.ravel())

#     #     print(xx_yy)

#     #     Z = clf.decision_function(xx_yy)
#     #     Z = Z.reshape(xx.shape)
#     #     plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#     #                 linestyles=['--', '-', '--'])
#     #     plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
#     #                 linewidth=1, facecolors='none', edgecolors='k')
#     #     plt.title("C=" + str(C))

#     # plt.tight_layout()
#     # plt.show()





#     # # assign training inputs and training targets to dataframe
#     # inputs_df = pd.DataFrame(inputs_test, columns=input_cols)

#     # # prepare X_train by reducing to 2d (for graphing purposes)
#     # pca = PCA(n_components = 2)
#     # X_train = pca.fit_transform(inputs_df)
#     # print(X_train)

#     # # prepare y_train by transforming from array of floats to array of integers
#     # y_train = np.int64(targets_test)

#     # # fit model
#     # clf.fit(X_train, y_train)

#     # # plot
#     # plot_decision_regions(X_train, y_train, clf=clf, legend=2)
#     # plt.xlabel(inputs_df.columns[0], size=14)
#     # plt.ylabel(inputs_df.columns[1], size=14)
#     # plt.title('SVM Decision Region Boundary', size=16)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA

def plot_kernels(inputs_train, targets_train, inputs_test, targets_test, input_cols, best_reg_param):
    """
    Function to aid exploratory analysis for future works, plots different different kernels. 
    """

    # define model
    clf = LinearSVC(random_state=42, dual=False, C=0.1)

    # assign training inputs and training targets to dataframe
    inputs_df = pd.DataFrame(inputs_train, columns=input_cols)

    # prepare X_train by reducing to 2d (for graphing purposes)
    pca = PCA(n_components = 2)
    X_train = pca.fit_transform(inputs_df)
    print(X_train)

    # prepare y_train by transforming from array of floats to array of integers
    y_train = np.int64(targets_train)

    # fit model
    clf.fit(X_train, y_train)

    # plot
    plot_decision_regions(X_train, y_train, clf=clf, legend=2)
    plt.xlabel(inputs_df.columns[0], size=14)
    plt.ylabel(inputs_df.columns[1], size=14)
    plt.title('SVM Decision Region Boundary', size=16)
    plt.show()



    # define model
    clf = LinearSVC(random_state=42, dual=False, C=0.1)

    # assign training inputs and training targets to dataframe
    inputs_df = pd.DataFrame(inputs_test, columns=input_cols)

    # prepare X_train by reducing to 2d (for graphing purposes)
    pca = PCA(n_components = 2)
    X_train = pca.fit_transform(inputs_df)
    print(X_train)

    # prepare y_train by transforming from array of floats to array of integers
    y_train = np.int64(targets_test)

    # fit model
    clf.fit(X_train, y_train)

    # plot
    plot_decision_regions(X_train, y_train, clf=clf, legend=2)
    plt.xlabel(inputs_df.columns[0], size=14)
    plt.ylabel(inputs_df.columns[1], size=14)
    plt.title('SVM Decision Region Boundary', size=16)

    # # assign training inputs and training targets to dataframe
    # # inputs_df = pd.DataFrame(inputs_train, columns=input_cols)
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # df = pd.DataFrame(inputs_test, columns=input_cols)
    # print(df)

    # # Initializing Classifiers
    # clf1 = SVC(random_state=42,kernel='linear')
    # clf2 = SVC(random_state=42,kernel='rbf')
    # clf3 = SVC(random_state=42, probability=True)
    # eclf = SVC(random_state=42,kernel='poly')

    # # eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[2, 1, 1], voting='soft')

    # # Choosing which features to plot
    # X_train = inputs_test[:, [2, 4]]

    # # assign training targets as integer array 
    # y_train = np.int64(targets_test)

    # # Plotting Decision Regions
    # gs = gridspec.GridSpec(2, 2)
    # fig = plt.figure(figsize=(10, 8))

    # # assign labels
    # labels = ['Logistic Regression',
    #         'Random Forest',
    #         'Linear SVM',
    #         'Ensemble']

    # # loop through classifiers, labels and grid space
    # for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
    #                         labels,
    #                         itertools.product([0, 1],
    #                         repeat=2)):
    #     clf.fit(X_train, y_train)
    #     ax = plt.subplot(gs[grd[0], grd[1]])
    #     fig = plot_decision_regions(X=X_train, y=y_train,
    #                                 clf=clf, legend=2)
    #     plt.title(lab)