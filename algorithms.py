import matplotlib.pyplot as plt
import numpy as np
import itertools
import keras

from preprocessing import Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.losses import binary_crossentropy ,categorical_crossentropy
from keras.optimizers import Adam




# Creates and shows confusion matrix

def confmat(classes, y_pred, y_test, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_test, y_pred)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
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


class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def run_cnn(X_train, X_test, y_train, y_test):

    batch_size = 20
    num_classes = 2
    epochs = 20
    history = AccuracyHistory()

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_shape = (293, 293, 1)


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(lr=0.01),
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[history])
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(range(1, 11), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()



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
X, y_gender = np.asarray(X), np.asarray(y_gender)
X = X.reshape(X.shape[0], 293, 293, 1)
#X = prp.vectorize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y_gender, test_size=0.2, random_state=2, stratify=y_gender)
run_cnn(X_train, X_test, y_train, y_test)

# Define classifiers

# Logistic Regression
"""
lr = LogisticRegression(C=1.0, random_state=2)
svm = SVC(C=1.0, kernel='linear', random_state=9)

kfold = StratifiedKFold(n_splits=10, random_state=2)
scores = []
X, y_gender = np.asarray(X), np.asarray(y_gender)
for k, (train, test) in enumerate(kfold.split(X, y_gender)):
    svm.fit(X[train], y_gender[train])
    score = svm.score(X[test], y_gender[test])
    print("Fold ", k, " accuracy: %.3f" % score)
    scores.append(score)
print("SVM accuracy: %.3f" % np.mean(scores))
"""

"""
print("Age classification: ")


run('age', 9, 'logreg')
run('age', 9, 'knn')
run('age', 9, 'svm')
run('age', 9, 'forest')

print("Gender Classification: ")

run('gender', 9, 'logreg')
run('gender', 9, 'knn')
run('gender', 9, 'svm')
run('gender', 9, 'forest')

"""




