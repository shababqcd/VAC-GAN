import numpy as np
import scipy.io
import pickle
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

DATA = scipy.io.loadmat('test_batch.mat')
images = DATA['data']
images = images.reshape(-1, 3, 32, 32)
labels = np.squeeze(DATA['labels'])

meta  = scipy.io.loadmat('batches.meta.mat')
meta = meta['label_names']
#netBestGASAClassifierCFAR10_DCGANAttempt2_Epoch
#netBestJustClassifierCFAR10_Attempt4
ACC = np.array([])
#for ii in range(100):
with open('netBestGASAClassifierCFAR10_DCGANAttempt2_2_Epoch33.pickle','rb') as handle:
    nno = pickle.load(handle)

Inputs = T.tensor4('inputs')
GT = T.vector('GT')
Outputs = lasagne.layers.get_output(nno,Inputs)
ClassesOut = T.argmax(Outputs, axis=1)
TestAcc = T.mean(T.eq(T.argmax(Outputs, axis=1), GT),
                  dtype=theano.config.floatX)

TestFunc = theano.function([Inputs,GT],[ClassesOut,TestAcc])
acc = 0
counter = 0
OUTNUM = np.array([])
for batches in iterate_minibatches(images,labels,100):
    InputsNum , LabelsNum = batches
    InputsNum = np.float32(InputsNum)/np.float32(255)*2.0-1.0
    OutNum , AccNum = TestFunc(InputsNum,LabelsNum)
    acc+=AccNum
    OUTNUM = np.append(OUTNUM,OutNum,axis=0)
    counter +=1
    print counter
print acc/counter
ACC = np.append(ACC,acc/counter)
cnf_matrix = confusion_matrix(labels, OUTNUM)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
                      title='Confusion matrix, without normalization')
plt.show()
print 'yup'

