#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example employing Lasagne for digit generation using the MNIST dataset and
Deep Convolutional Generative Adversarial Networks
(DCGANs, see http://arxiv.org/abs/1511.06434).
It is based on the MNIST example in Lasagne:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
Note: In contrast to the original paper, this trains the generator and
discriminator at once, not alternatingly. It's easy to change, though.
Jan Schlüter, 2015-12-16
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import pickle
import scipy.io
#from MODELSgs10ClassesIncluding28x28 import ClassifierNetwork28x28

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.
class MinibatchLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_kernels, dim_per_kernel=5, theta=lasagne.init.Normal(0.05),
                 log_weight_scale=lasagne.init.Constant(0.), b=lasagne.init.Constant(-1.), **kwargs):
        super(MinibatchLayer, self).__init__(incoming, **kwargs)
        self.num_kernels = num_kernels
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_kernels, dim_per_kernel), name="theta")
        self.log_weight_scale = self.add_param(log_weight_scale, (num_kernels, dim_per_kernel), name="log_weight_scale")
        self.W = self.theta * (T.exp(self.log_weight_scale) / T.sqrt(T.sum(T.square(self.theta), axis=0))).dimshuffle(
            'x', 0, 1)
        self.b = self.add_param(b, (num_kernels,), name="b")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]) + self.num_kernels)

    def get_output_for(self, input, init=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.tensordot(input, self.W, [[1], [0]])
        abs_dif = (T.sum(abs(activation.dimshuffle(0, 1, 2, 'x') - activation.dimshuffle('x', 1, 2, 0)), axis=2)
                   + 1e6 * T.eye(input.shape[0]).dimshuffle(0, 'x', 1))

        if init:
            mean_min_abs_dif = 0.5 * T.mean(T.min(abs_dif, axis=2), axis=0)
            abs_dif /= mean_min_abs_dif.dimshuffle('x', 0, 'x')
            self.init_updates = [
                (self.log_weight_scale, self.log_weight_scale - T.log(mean_min_abs_dif).dimshuffle(0, 'x'))]

        f = T.sum(T.exp(-abs_dif), axis=2)

        if init:
            mf = T.mean(f, axis=0)
            f -= mf.dimshuffle('x', 0)
            self.init_updates.append((self.b, -mf))
        else:
            f += self.b.dimshuffle('x', 0)

        return T.concatenate([input, f], axis=1)


def FirstTenSeedsGen(targets):
    SEEDt = np.float32(np.zeros([len(targets),10]))
    for ii in range(len(targets)):
        target = targets[ii]
        if target == 0:
            seedT = [np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand()]
        elif target == 1:
            seedT = [-np.random.rand(),np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand()]
        elif target == 2:
            seedT = [-np.random.rand(),-np.random.rand(),np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand()]
        elif target == 3:
            seedT = [-np.random.rand(),-np.random.rand(),-np.random.rand(),np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand()]
        elif target == 4:
            seedT = [-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand()]
        elif target == 5:
            seedT = [-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand()]
        elif target == 6:
            seedT = [-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand()]
        elif target == 7:
            seedT = [-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),np.random.rand(),-np.random.rand(),-np.random.rand()]
        elif target == 8:
            seedT = [-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),np.random.rand(),-np.random.rand()]
        else:
            seedT = [-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),-np.random.rand(),np.random.rand()]

        SEEDt[ii,:] = seedT
    return SEEDt
'''def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test'''

def load_dataset():
    DATA = scipy.io.loadmat('data_batch_1.mat')
    Images1 = DATA['data']
    Images1 = Images1.reshape(-1, 3, 32, 32)
    Labels1 = DATA['labels']

    DATA = scipy.io.loadmat('data_batch_2.mat')
    Images2 = DATA['data']
    Images2 = Images2.reshape(-1, 3, 32, 32)
    Labels2 = DATA['labels']

    DATA = scipy.io.loadmat('data_batch_3.mat')
    Images3 = DATA['data']
    Images3 = Images3.reshape(-1, 3, 32, 32)
    Labels3 = DATA['labels']

    DATA = scipy.io.loadmat('data_batch_4.mat')
    Images4 = DATA['data']
    Images4 = Images4.reshape(-1, 3, 32, 32)
    Labels4 = DATA['labels']

    DATA = scipy.io.loadmat('data_batch_5.mat')
    Images5 = DATA['data']
    Images5 = Images5.reshape(-1, 3, 32, 32)
    Labels5 = DATA['labels']

    Images = np.concatenate((Images1, Images2, Images3, Images4, Images5), axis=0)
    Labels = np.concatenate((Labels1, Labels2, Labels3, Labels4, Labels5), axis=0)
    return Images, Labels


# ##################### Build the neural network model #######################
# We create two models: The generator and the discriminator network. The
# generator needs a transposed convolution layer defined first.

'''class Deconv2DLayer(lasagne.layers.Layer):

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
        return self.nonlinearity(conved)'''

CONTROLDIM = 10
batch_size = 16
def build_generator(input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm , TransposedConv2DLayer
    from lasagne.nonlinearities import sigmoid
    # input: 100dim
    GenL1 = InputLayer(shape=(None, CONTROLDIM), input_var=input_var)
    GenL2 = DenseLayer(GenL1 , num_units=384*4*4)
    GenL2Reshaped = ReshapeLayer(GenL2,shape=(batch_size,384,4,4))
    GenL3 = TransposedConv2DLayer(GenL2Reshaped,num_filters=192,filter_size=(5,5),stride=(2,2),crop='same',output_size=(8,8))
    GenL3BN = batch_norm(GenL3)
    GenL4 = TransposedConv2DLayer(GenL3BN,num_filters=96,filter_size=(5,5),stride=(2,2),crop='same',output_size=(16,16))
    GenL4BN = batch_norm(GenL4)
    outputLayer = TransposedConv2DLayer(GenL4BN,num_filters=3,filter_size=(5,5),stride=(2,2),nonlinearity=lasagne.nonlinearities.tanh,crop='same',output_size=(32,32))

    # fully-connected layer
    #layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    #layer = batch_norm(DenseLayer(layer, 128 * 7 * 7))
    #layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    # two fractional-stride convolutions
    #layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, pad=2))
    #layer = Deconv2DLayer(layer, 1, 5, stride=2, pad=2,
                         # nonlinearity=sigmoid)
    print("Generator output:", outputLayer.output_shape)
    return outputLayer

#kk = build_generator()

def DiscLayer(INCOMING,CHANNELS = 16 ,STRIDE = (1,1)):
    from lasagne.layers import Conv2DLayer, batch_norm, DropoutLayer
    output = Conv2DLayer(INCOMING,num_filters=CHANNELS,filter_size=(3,3),stride=STRIDE,pad='same',nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
    output = batch_norm(output)
    output = DropoutLayer(output)
    return output


def build_discriminator(input_var=None):
    #from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
     #                           DenseLayer, batch_norm)
    #from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer  # override
    #from lasagne.nonlinearities import LeakyRectify, sigmoid
    #lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    DiscL1 = lasagne.layers.InputLayer(shape=(None,3,32,32),input_var=input_var)
    DiscL2 = lasagne.layers.GaussianNoiseLayer(DiscL1,sigma=0.05)
    DiscL3 = lasagne.layers.Conv2DLayer(DiscL2,16,(3,3),stride=(2,2),pad='same',nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
    DiscL3DO = lasagne.layers.DropoutLayer(DiscL3)
    DiscL4 = DiscLayer(DiscL3DO,CHANNELS=32,STRIDE=(1,1))
    DiscL5 = DiscLayer(DiscL4,CHANNELS=64,STRIDE=(2,2))
    DiscL6 = DiscLayer(DiscL5,CHANNELS=128,STRIDE=(1,1))
    DiscL7 = DiscLayer(DiscL6,CHANNELS=256,STRIDE=(2,2))
    DiscL8 = DiscLayer(DiscL7,CHANNELS=512,STRIDE=(1,1))
    DiscL9 = MinibatchLayer(DiscL8,50,30)
    DiscOutput = lasagne.layers.DenseLayer(DiscL9,num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)

    #layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    # two convolutions
    #layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    #layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu))
    # fully-connected layer
    #layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer
    #layer = DenseLayer(layer, 1, nonlinearity=sigmoid)
    print("Discriminator output:", DiscOutput.output_shape)
    return DiscOutput


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

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


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.


num_epochs = 1000
initial_eta = 2e-4
alpha = .5
beta =.5
batch_size = 16
# Load the dataset
print("Loading data...")
X_train, y_train = load_dataset()

#X_train = np.delete(X_train,np.where(y_train==9),axis=0)
#y_train = np.delete(y_train,np.where(y_train==9),axis=0)

# Prepare Theano variables for inputs and targets
noise_var = T.matrix('noise')
input_var = T.tensor4('inputs')
target_varB = T.ivector(name='target_B')
#    target_var = T.ivector('targets')

# Create neural network model
print("Building model and compiling functions...")
generator = build_generator(noise_var)
discriminator = build_discriminator(input_var)

NetBL1 = lasagne.layers.InputLayer(shape=(None,3,32,32))
NetBL2 = lasagne.layers.Conv2DLayer(NetBL1,num_filters=128,filter_size=(5,5))
NetBL2BN = lasagne.layers.batch_norm(NetBL2)
NetBL2MP = lasagne.layers.MaxPool2DLayer(NetBL2BN,pool_size=(2,2))
NetBL3 = lasagne.layers.Conv2DLayer(NetBL2MP,num_filters=256,filter_size=(5,5))
NetBL3BN = lasagne.layers.batch_norm(NetBL3)
NetBL3MP = lasagne.layers.MaxPool2DLayer(NetBL3BN,pool_size=(2,2))
NetBL4 = lasagne.layers.Conv2DLayer(NetBL3MP,num_filters=512,filter_size=(5,5))
NetBL6 = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(NetBL4),num_units=512)
#NetBL7 = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(NetBL6),num_units=256)
#NetBL8 = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(NetBL7),num_units=256)
networkBOut = lasagne.layers.DenseLayer(NetBL6,num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
#networkBOut = ClassifierNetwork28x28(name='Classifier')

# Create expression for passing real data through the discriminator
real_out = lasagne.layers.get_output(discriminator)
# Create expression for passing fake data through the discriminator
fake_out = lasagne.layers.get_output(discriminator,
                                     lasagne.layers.get_output(generator))
# Classifier variables
NetB_all_input = T.concatenate([lasagne.layers.get_output(generator,inputs=noise_var),
                                input_var], axis=0)
#NetB_all_input = lasagne.layers.get_output(generator,inputs=noise_var)
NetB_output_all = T.add(lasagne.layers.get_output(networkBOut,inputs=NetB_all_input),np.finfo(np.float32).eps)
# Classifier losses
#NetB_loss_fake = lasagne.objectives.categorical_crossentropy(NetB_output_fake,targets_fake)
#NetB_loss_fake = NetB_loss_fake.mean()

NetB_loss_all = lasagne.objectives.categorical_crossentropy(NetB_output_all,target_varB)
NetB_loss_all = NetB_loss_all.mean()
# Create loss expressions
generator_loss = alpha * lasagne.objectives.binary_crossentropy(fake_out, 1).mean() + beta * NetB_loss_all
discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1)
                      + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()
NetB_loss_T = NetB_loss_all
# Create update expressions for training
generator_params = lasagne.layers.get_all_params(generator, trainable=True)
discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
NetB_params = lasagne.layers.get_all_params(networkBOut , trainable=True)

eta = theano.shared(lasagne.utils.floatX(initial_eta))
updates = lasagne.updates.adam(
    generator_loss, generator_params, learning_rate=eta, beta1=0.5)
updates.update(lasagne.updates.adam(
    discriminator_loss, discriminator_params, learning_rate=eta, beta1=0.5))
updates.update(lasagne.updates.nesterov_momentum(NetB_loss_T,NetB_params,learning_rate=0.01,momentum=0.9))
# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([noise_var, input_var,target_varB],
                           [(real_out > .5).mean(),
                            (fake_out < .5).mean(),lasagne.objectives.binary_crossentropy(fake_out, 1).mean(),
                            lasagne.objectives.categorical_crossentropy(NetB_output_all,target_varB).mean()],
                           updates=updates)

# Compile another function generating some data
gen_fn = theano.function([noise_var],
                         lasagne.layers.get_output(generator,
                                                   deterministic=True))

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
        #dummy = dummy + 1
        inputsNum, targetsNum = batch
        inputsNum = (np.float32(inputsNum) / np.float32(255))*2-1
        firstSeeds = FirstTenSeedsGen(targetsNum)
        seeds = np.float32(np.random.rand(batch_size, CONTROLDIM) * 2 - 1)
        seeds[:, 0:10] = firstSeeds
        targetsNum = np.squeeze(np.concatenate([targetsNum, targetsNum]))
        train_err += np.array(train_fn(seeds, inputsNum,targetsNum))
        train_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{}".format(train_err / train_batches))

    # And finally, we plot some generated data

    #with open('netBestDiscriminatorCFAR10_DCGANAttempt1Epoch' + str(
     #               epoch) + '.pickle', 'wb') as handle:
      #  print('saving the model Discriminator....')
       # pickle.dump(discriminator, handle)
    with open('netBestGASAClassifierCFAR10_DCGANAttempt2_2_Epoch'+ str(
                    epoch) + '.pickle', 'wb') as handle:
        print('saving the model Classifier....')
        pickle.dump(networkBOut, handle)

    with open('netBestGASAGeneratorCFAR10_DCGANAttempt2_2_Epoch' + str(
                    epoch) + '.pickle', 'wb') as handle:
        print('saving the model Generator....')
        pickle.dump(generator, handle)
    # After half the epochs, we start decaying the learn rate towards zero
    if epoch >= num_epochs // 2:
        progress = float(epoch) / num_epochs
        eta.set_value(lasagne.utils.floatX(initial_eta * 2 * (1 - progress)))

# Optionally, you could now dump the network weights to a file like this:
#np.savez('mnist_gen.npz', *lasagne.layers.get_all_param_values(generator))
#np.savez('mnist_disc.npz', *lasagne.layers.get_all_param_values(discriminator))
#
# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)


