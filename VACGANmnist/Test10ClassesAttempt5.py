import sys
import numpy as np
import os
import scipy.misc
import scipy.io
import theano
import theano.tensor as T
import lasagne
import time
import pickle
import matplotlib.pyplot as plt
from MODELSgs10ClassesIncluding28x28 import ClassifierNetwork28x28
class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Orthogonal(),
                                (self.input_shape[1], num_filters) + self.filter_size,
                                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                                (num_filters,),
                                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i * s - 2 * p + f - 1
                      for i, s, p, f in zip(input_shape[2:],
                                            self.stride,
                                            self.pad,
                                            self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)

def FirstTenSeedsGenTest(targets):
    SEEDt = np.float32(np.zeros([len(targets),10]))
    a=.7
    b=.3
    for ii in range(len(targets)):
        target = targets[ii]
        if target == 0:
            seedT = [(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b)]
        elif target == 1:
            seedT = [-(np.random.rand()*a+b),(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b)]
        elif target == 2:
            seedT = [-(np.random.rand()*a+b),-(np.random.rand()*a+b),(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b)]
        elif target == 3:
            seedT = [-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b)]
        elif target == 4:
            seedT = [-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b)]
        elif target == 5:
            seedT = [-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b)]
        elif target == 6:
            seedT = [-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b)]
        elif target == 7:
            seedT = [-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b)]
        elif target == 8:
            seedT = [-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),(np.random.rand()*a+b),-(np.random.rand()*a+b)]
        else:
            seedT = [-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),-(np.random.rand()*a+b),(np.random.rand()*a+b)]

        SEEDt[ii,:] = seedT
    return SEEDt
batch_size = 16
CONTROLDIM = 10
targetsNum = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7]
with open('netBestGASAGeneratorSingleGenMNIST_DCGANAttempt5_10ClassesEpoch55.pickle','rb') as handle:
    nno = pickle.load(handle)

firstTwoSeeds = FirstTenSeedsGenTest(targetsNum[0:batch_size])
seeds = np.float32(np.random.rand(batch_size, CONTROLDIM)*2-1)
seeds[:,0:10] = firstTwoSeeds

outputs = lasagne.layers.get_output(nno,seeds)
imageOut = outputs.eval()
f2, ((axx1, axx2 , axx3 , axx4), (axx5, axx6, axx7, axx8), (axx9, axx10, axx11,axx12),(axx13,axx14,axx15,axx16)) = plt.subplots(4, 4, sharey=True)
axx1.imshow(imageOut[0 , 0 , : , :],cmap='gray')
axx2.imshow(imageOut[1 , 0 , : , :],cmap='gray')
axx3.imshow(imageOut[2 , 0 , : , :],cmap='gray')
axx4.imshow(imageOut[3 , 0 , : , :],cmap='gray')
axx5.imshow(imageOut[4 , 0 , : , :],cmap='gray')
axx6.imshow(imageOut[5 , 0 , : , :],cmap='gray')
axx7.imshow(imageOut[6 , 0 , : , :],cmap='gray')
axx8.imshow(imageOut[7 , 0 , : , :],cmap='gray')
axx9.imshow(imageOut[8 , 0 , : , :],cmap='gray')
axx10.imshow(imageOut[9 , 0 , : , :],cmap='gray')
axx11.imshow(imageOut[10 , 0 , : , :],cmap='gray')
axx12.imshow(imageOut[11 , 0 , : , :],cmap='gray')
axx13.imshow(imageOut[12 , 0 , : , :],cmap='gray')
axx14.imshow(imageOut[13 , 0 , : , :],cmap='gray')
axx15.imshow(imageOut[14 , 0 , : , :],cmap='gray')
axx16.imshow(imageOut[15 , 0 , : , :],cmap='gray')
plt.show()
print 'yup'