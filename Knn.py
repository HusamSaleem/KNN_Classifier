import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
import numpy as np
from sklearn import metrics

def split_input_data(data):
    x = [[]] # Attributes
    y = [] # labels

    for c in data.itertuples():
        attribute = [c[1], c[2]]
        x.append(attribute)
        y.append(c[3])
    
    x.pop(0)
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    return X_train, X_test, y_train, y_test

def display_contours(X_train, y_train, classifier, number_of_neighbors):
    plot_decision_regions(np.asarray(X_train), np.asarray(y_train), clf=classifier)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Knn with K=%d' % (number_of_neighbors))
    plt.show()

def knn(nneighbors, X_train, y_train, X_test):
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X_train, y_train)

    predicted_y = clf.predict(X_test)
    display_contours(X_train, y_train, clf, nneighbors)
    return predicted_y

def evaluateknn(y_predicted, y_test):
    confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
    class_report = metrics.classification_report(y_test, y_predicted)
    print('Confusion Matrix:\n', confusion_matrix)
    print('\nClassification Report:\n', class_report)

if __name__ == "__main__":
    input_data = pd.read_csv("A1-inputData.csv")
    X_train, X_test, y_train, y_test = split_input_data(input_data)

    predicted_y = knn(3, X_train, y_train, X_test)
    evaluateknn(predicted_y, y_test)