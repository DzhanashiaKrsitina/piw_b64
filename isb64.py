from sklearn.model_selection import train_test_split
from sklearn import linear_model, neighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt


def get_features(s):  # признаки для классификации
    s = s[:32]
    return [sum(c.isdigit() for c in s) / len(s), sum(c.islower() for c in s) / len(s), s.count('=')]


def create_xy(fill):
    X, y = [], []
    for row in fill:
        y.append(int(row[0]))
        X.append(get_features(row[1:-1]))
    return X, y


fill = open('dataset.txt', 'r', encoding='utf-8')
X, y = create_xy(fill)
fill.close()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifiers = [
    #linear_model.LinearRegression(),
    #linear_model.Ridge(alpha=.5),
    #linear_model.Lasso(alpha=0.1),
    #linear_model.BayesianRidge(),
    #neighbors.KNeighborsClassifier(2, weights='uniform'),
    #SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=5, max_features=1),
    #MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis()
    ]

data = []
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if isinstance(y_pred[0], float):
        y_pred = list(map(round, y_pred))
    correct = 0
    for x in range(len(X_test)):
        if y_pred[x] == y_test[x]:
            correct += 1
    print(str(clf).partition("(")[0])
    print('accuracy ' + str(correct / len(y_pred)))
    data.append([str(clf).partition("(")[0], str("%.2f" %(100 * correct / len(y_pred))) + ' %'])
data = sorted(data)
data = np.asarray(data)

plt.axis('off')
plt.table(cellText=data, colLabels=("Classifier", "Accuracy"), loc='center')
plt.show()
