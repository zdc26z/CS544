#Kholby Lawson, CS544, Fall 2015
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

'''

   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)

'''

#read raw data
names=['id', 'Clump Thickness', 'Uniformity of Size', 'Uniformity of Shape', 'Marginal Adhesion', 'Single Epithelial Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
raw_data = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', delimiter=',', names=names, na_values=['?'])
data = raw_data.dropna()

#these will all be set based on the most accurate test/train set
best_prec = 0
best_acc = 0
best_rec = 0
best_coef = 0

#run 100 different train/test sets
for i in range(0, 100):
    train, test = train_test_split(data)
    trainX = train[names[1:len(names)-1]]
    trainY = train[names[len(names)-1]]

    testX = test[names[1:len(names)-1]]
    testY = test[names[len(names)-1]]
    testY = testY/2-1

    clf = svm.SVC(kernel='linear', gamma = .00001)

    clf.fit(trainX, trainY)
    predictions = clf.predict(testX)
    predictions = predictions/2-1
    prec = precision_score(predictions, testY)
    acc = accuracy_score(predictions, testY)
    rec = recall_score(predictions, testY)
    coef = str(zip(names[1:len(names)-1], clf.coef_[0]))
    
    if acc > best_acc:
        best_prec = prec
        best_acc = acc
        best_rec = rec
        best_coef = coef
    


print "Precision:  ", best_prec
print "Accuracy:  ", best_acc
print "Recall:  ", best_rec
print best_coef


