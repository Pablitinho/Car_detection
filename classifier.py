
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score

clf =0

def train_svm_classifier(data,label):

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(data, label)

    pred = clf.predict(data)
    acc = accuracy_score(pred, label)
    print("Training Accuracy: ", acc*100, "%")

    return clf

def evaluate_svm(clf, pattern):

    return clf.predict(pattern)


