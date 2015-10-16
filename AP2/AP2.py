from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

CENTER_X = .5
CENTER_Y = .5
RADIUS = .4

points =  np.random.rand(1000,2)
alldata = ClassificationDataSet(2, 1, 2)
figure('Training data')
clf()
ioff()
for point in points:
    if (point[0] - CENTER_X)**2 + (point[1]-CENTER_Y)**2 <= RADIUS**2:
        klass = 0
        #plot(point[0], point[1], 'bo')
    else:
        klass = 1
        #plot(point[0], point[1], 'ro')
    alldata.addSample(point, klass)

tstdata, trndata = alldata.splitWithProportion(.25)
tstdata._convertToOneOfMany()
trndata._convertToOneOfMany()

inC, _ = where(trndata['class'] == 0)
outC, _ = where(trndata['class'] == 1)
plot(trndata['input'][inC, 0], trndata['input'][inC, 1], 'bo')
plot(trndata['input'][outC, 0], trndata['input'][outC, 1], 'ro')

fnn = buildNetwork(trndata.indim, 9, trndata.outdim, outclass=SoftmaxLayer, bias=True)
trainer = BackpropTrainer(fnn, dataset=trndata, momentum = .99)


trainer.trainUntilConvergence(maxEpochs=1000)
trnresult = percentError( trainer.testOnClassData(), 
                         trndata['class'] )
tstresult = percentError( trainer.testOnClassData( 
   dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs, \
    "  train error: %5.2f%%" % trnresult, \
    "  test error: %5.2f%%" % tstresult


results = fnn.activateOnDataset(dataset=tstdata)
out = results.argmax(axis=1)

figure('Test data')
for ind, tc in enumerate(tstdata['class']):
    if tc==out[ind]:
        #match.append(ind)
        if tc==0:
            plot(tstdata['input'][ind, 0], tstdata['input'][ind, 1], 'bo')
        else:
            plot(tstdata['input'][ind, 0], tstdata['input'][ind, 1], 'ro')
    else:
        #nomatch.append(ind)
        plot(tstdata['input'][ind, 0], tstdata['input'][ind, 1], 'ko')

print "Precision:  ", precision_score(tstdata['class'], out)
print "Accuracy:  ", accuracy_score(tstdata['class'], out)
print "Recall:  ", recall_score(tstdata['class'], out)
show()
    

