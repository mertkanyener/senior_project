import matplotlib.pyplot as plt
import numpy as np
import itertools

from preprocessing import Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



# Creates and shows confusion matrix

def confmat(classes, y_pred, y_test, classifier, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_test, y_pred)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(classifier)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def run(option, random_state, classifier):
    # %20 test %80 train split
    X_train, X_test, y_train, y_test = preprocessing(option)
    # Get class labels
    if option == 'gender':
        classes = ['female', 'male']
    else:
        classes = ['Age 18-29', 'Age 30-49', 'Age 50-69', 'Age 70-94']

    if classifier == 'logreg':

        pipe_lr = make_pipeline(
                                LogisticRegression(C=1.0, random_state=random_state))

        pipe_lr.fit(X_train, y_train)
        y_pred = pipe_lr.predict(X_test)
        print('Logistic Regression:')
        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
        plt.figure()
        confmat(classes, y_pred, y_test)
        plt.show()

    elif classifier == 'svm':

        pipe_svm = make_pipeline(SVC(kernel='linear', random_state=random_state))
        pipe_svm.fit(X_train, y_train)
        y_pred = pipe_svm.predict(X_test)
        print('Support Vector Machine:')
        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
        plt.figure()
        confmat(classes, y_pred, y_test)
        plt.show()

    elif classifier == 'forest':

        pipe_forest = make_pipeline(RandomForestClassifier(criterion='gini', n_estimators=500,
                                                           n_jobs=2, random_state=random_state))
        pipe_forest.fit(X_train, y_train)
        y_pred = pipe_forest.predict(X_test)
        print('Random Forest:')
        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
        plt.figure()
        confmat(classes, y_pred, y_test)
        plt.show()

    elif classifier == 'knn':

        pipe_knn = make_pipeline(KNeighborsClassifier(n_neighbors=5))
        pipe_knn.fit(X_train, y_train)
        y_pred = pipe_knn.predict(X_test)
        print('K-Nearest Neighbors:')
        plt.figure()
        print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
        confmat(classes, y_pred, y_test)
        plt.show()

    else:
        print('Wrong Classifier Name!!')
        print("Logsitic Regression = 'logreg'")
        print("Support Vector Machine = 'svm'")
        print("Random Forest Classifier = 'forest'")
        print("K-Nearest Neighbors Classifier = 'knn'")


"""
dir1 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_70-94_Neutral_bmp/male"
dataset = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]


"""
dir1 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_aligned_cut/BW_age_70-94_Neutral_bmp/male"
dataset = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]

prp = Preprocessing()
X, y_gender, y_age = prp.read_data(dataset)
X = prp.vectorize(X)
X, y_gender = np.asarray(X), np.asarray(y_gender)
y_age = np.asarray(y_age)
#X = X.reshape(X.shape[0], 293, 293, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y_age, test_size=0.2, random_state=2, stratify=y_age)

classes_age = ['Age 18-29', 'Age 30-49', 'Age 50-69', 'Age 70-94']
classes_gender = ['Female', 'Male']
"""
# Age Estimation

print("Age Classification: ")

rnd = 1
# Logistic Regression
lr = LogisticRegression(C=1.0, random_state=rnd)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('Logistic Regression:')
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
plt.figure()
confmat(classes_age, y_pred, y_test, 'Logistic Regression')
plt.show()

# Random Forest Classifier

forest = RandomForestClassifier(criterion='gini', n_estimators=500, n_jobs=2, random_state=rnd)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print('Random Forest:')
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
plt.figure()
confmat(classes_age, y_pred, y_test, 'Random Forest')
plt.show()

# SVM 

svm = SVC(kernel='linear', random_state=rnd)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('SVM:')
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
plt.figure()
confmat(classes_age, y_pred, y_test, 'SVM')
plt.show()

# KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('KNN:')
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
plt.figure()
confmat(classes_age, y_pred, y_test, 'KNN')
plt.show()

"""
"""
# Gender Classification

lr = LogisticRegression(C=1.0, random_state=rnd)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('Logistic Regression:')
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
plt.figure()
confmat(classes_gender, y_pred, y_test, 'Logistic Regression')
plt.show()

forest = RandomForestClassifier(criterion='gini', n_estimators=500, n_jobs=2, random_state=rnd)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print('Random Forest:')
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
plt.figure()
confmat(classes_gender, y_pred, y_test, 'Random Forest')
plt.show()

svm = SVC(kernel='linear', random_state=rnd)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('SVM:')
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
plt.figure()
confmat(classes_gender, y_pred, y_test, 'SVM')
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('KNN:')
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
plt.figure()
confmat(classes_gender, y_pred, y_test, 'KNN')
plt.show()

"""
"""
print()

for k, (train, test) in enumerate(kfold.split(X, y_gender)):
    svm.fit(X[train], y_gender[train])
    score = svm.score(X[test], y_gender[test])
    print("Fold ", k+1, " accuracy: %.3f" % score)
    scores.append(score)
print("SVM accuracy: %.3f" % np.mean(scores))

for k, (train, test) in enumerate(kfold.split(X, y_gender)):
    lr.fit(X[train], y_gender[train])
    score = lr.score(X[test], y_gender[test])
    print("Fold ", k, " accuracy: %.3f" % score)
    scores.append(score)
print("Logistic Regression accuracy: %.3f" % np.mean(scores))

for k, (train, test) in enumerate(kfold.split(X, y_gender)):
    forest.fit(X[train], y_gender[train])
    score = forest.score(X[test], y_gender[test])
    print("Fold ", k, " accuracy: %.3f" % score)
    scores.append(score)
print("Random Forest accuracy: %.3f" % np.mean(scores))

for k, (train, test) in enumerate(kfold.split(X, y_gender)):
    knn.fit(X[train], y_gender[train])
    score = knn.score(X[test], y_gender[test])
    print("Fold ", k, " accuracy: %.3f" % score)
    scores.append(score)
print("KNN accuracy: %.3f" % np.mean(scores))
"""

# K fold Age Classification

lr = LogisticRegression(C=1.0, random_state=9)
svm = SVC(C=1.0, kernel='linear', random_state=9)
forest = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=9)
knn = KNeighborsClassifier(n_neighbors=7)

kfold = StratifiedKFold(n_splits=10, random_state=9)
scores = []
X, y_gender, y_age = np.asarray(X), np.asarray(y_gender), np.asarray(y_age)

print("Age Classification kfold: ")

for k, (train, test) in enumerate(kfold.split(X, y_age)):
    svm.fit(X[train], y_age[train])
    score = svm.score(X[test], y_age[test])
    print("Fold ", k, " accuracy: %.3f" % score)
    scores.append(score)
print("SVM accuracy: %.3f" % np.mean(scores))

for k, (train, test) in enumerate(kfold.split(X, y_age)):
    lr.fit(X[train], y_age[train])
    score = lr.score(X[test], y_age[test])
    print("Fold ", k, " accuracy: %.3f" % score)
    scores.append(score)
print("Logistic Regression accuracy: %.3f" % np.mean(scores))

for k, (train, test) in enumerate(kfold.split(X, y_age)):
    forest.fit(X[train], y_age[train])
    score = forest.score(X[test], y_age[test])
    print("Fold ", k, " accuracy: %.3f" % score)
    scores.append(score)
print("Random Forest accuracy: %.3f" % np.mean(scores))

for k, (train, test) in enumerate(kfold.split(X, y_age)):
    knn.fit(X[train], y_age[train])
    score = knn.score(X[test], y_age[test])
    print("Fold ", k, " accuracy: %.3f" % score)
    scores.append(score)
print("KNN accuracy: %.3f" % np.mean(scores))










