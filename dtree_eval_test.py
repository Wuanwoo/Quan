'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score


def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
    '''
    
    # Load Data
    filename = 'D:/Assignment1/hw1_skeleton/data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    
    meanDecisionTreeAccuracies = []
    meanDecisionStumpAccuracies = []
    meanDT3Accuracies = []

    # 100 trials of 10-fold cross-validation
    for trial in range(100):
        np.random.seed(trial)
        np.random.shuffle(data)

        accuracies_decision_tree = []
        accuracies_decision_stump = []
        accuracies_dt3 = []

        # 10-fold cross-validation
        for fold in range(10):
            start = fold * 27
            end = (fold + 1) * 27

            # Split the data into training and test sets
            Xtrain = np.concatenate((data[:start, 1:], data[end:, 1:]))
            ytrain = np.concatenate((np.array([data[:start, 0]]).T, np.array([data[end:, 0]]).T))
            Xtest = data[start:end, 1:]
            ytest = np.array([data[start:end, 0]]).T

            # Train the decision tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(Xtrain, ytrain)

            # Output predictions on the test set
            y_pred = clf.predict(Xtest)
            accuracies_decision_tree.append(accuracy_score(ytest, y_pred))

            # Train the decision stump (1-level decision tree)
            clf_stump = tree.DecisionTreeClassifier(max_depth=1)
            clf_stump = clf_stump.fit(Xtrain, ytrain)

            # Output predictions on the test set
            y_pred_stump = clf_stump.predict(Xtest)
            accuracies_decision_stump.append(accuracy_score(ytest, y_pred_stump))

            # Train the 3-level decision tree
            clf_dt3 = tree.DecisionTreeClassifier(max_depth=3)
            clf_dt3 = clf_dt3.fit(Xtrain, ytrain)

            # Output predictions on the test set
            y_pred_dt3 = clf_dt3.predict(Xtest)
            accuracies_dt3.append(accuracy_score(ytest, y_pred_dt3))

        # Store mean accuracy for each classifier for the current trial
        meanDecisionTreeAccuracies.append(np.mean(accuracies_decision_tree))
        meanDecisionStumpAccuracies.append(np.mean(accuracies_decision_stump))
        meanDT3Accuracies.append(np.mean(accuracies_dt3))

    # Compute mean and standard deviation of accuracies across all trials
    meanDecisionTreeAccuracy = np.mean(meanDecisionTreeAccuracies)
    stddevDecisionTreeAccuracy = np.std(meanDecisionTreeAccuracies)
    meanDecisionStumpAccuracy = np.mean(meanDecisionStumpAccuracies)
    stddevDecisionStumpAccuracy = np.std(meanDecisionStumpAccuracies)
    meanDT3Accuracy = np.mean(meanDT3Accuracies)
    stddevDT3Accuracy = np.std(meanDT3Accuracies)

    # Make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanDecisionStumpAccuracy
    stats[1, 1] = stddevDecisionStumpAccuracy
    stats[2, 0] = meanDT3Accuracy
    stats[2, 1] = stddevDT3Accuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
