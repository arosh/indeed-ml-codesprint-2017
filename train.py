import parser
import pandas
import re
import bm25
import sklearn.naive_bayes
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.externals
import sklearn.tree
import sklearn.ensemble
import sklearn.base
import sklearn.svm
import scipy
import warnings

def predict_train(clf, params, X_train, y_train):
    y_pred = pandas.DataFrame(columns=y_train.columns)
    estms = []
    for column in y_train.columns:
        cv = sklearn.model_selection.RandomizedSearchCV(
                clf,
                params,
                scoring='f1',
                cv=10,
                n_iter=20,
                n_jobs=-1,
                verbose=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.fit(X_train, y_train[column])
        print(cv.best_score_, cv.best_params_)
        y_pred[column] = sklearn.model_selection.cross_val_predict(
                cv.best_estimator_,
                X_train,
                y_train[column],
                cv=10,
                n_jobs=-1)
        estms.append(cv.best_estimator_)
    return y_pred, estms

def predict_random_forest(clf_origin, X_train, y_train):
    y_pred = pandas.DataFrame(columns=y_train.columns)
    estms = []
    for column in y_train.columns:
        print(column)
        clf = sklearn.base.clone(clf_origin)
        y_pred[column] = sklearn.model_selection.cross_val_predict(
                clf,
                X_train,
                y_train[column],
                cv=10,
                n_jobs=-1)
        estms.append(clf)
    return y_pred, estms

def f1score(y_true, y_pred):
    STN = 0
    SFN = 0
    SFP = 0
    STP = 0
    for column in y_true.columns:
        cm = sklearn.metrics.confusion_matrix(y_true[column], y_pred[column])
        STN += cm[0,0]
        SFN += cm[1, 0]
        SFP += cm[0, 1]
        STP += cm[1, 1]
    P = STP / (STP + SFP)
    R = STP / (STP + SFN)
    F1 = 2*P*R/(P+R)
    return F1

def main():
    X_train = sklearn.externals.joblib.load('X_train.pkl')
    y_train = sklearn.externals.joblib.load('y_train.pkl')

    # clf = sklearn.svm.LinearSVC(dual=False, penalty='l1')
    # params = {
    #     'C': 10**scipy.linspace(-5, 5, 100),
    # }
    # X_train = bm25.BM25Transformer().fit_transform(X_train)
    # clf = sklearn.naive_bayes.MultinomialNB()
    # clf = sklearn.naive_bayes.BernoulliNB()
    # params = {
    #     'alpha': 10**scipy.linspace(-7, 0, 1000),
    # }
    clf = sklearn.tree.DecisionTreeClassifier()
    # clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_features=None)
    params = {
        'max_depth': scipy.arange(1, 21),
    }
    y_pred, estms = predict_train(clf, params, X_train, y_train)
    # clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=20, max_features=None)
    # y_pred, estms = predict_random_forest(clf, X_train, y_train)
    print('f1score =', f1score(y_train, y_pred))
    sklearn.externals.joblib.dump(estms, 'estms.pkl')

if __name__ == '__main__':
    main()
