import numpy as np
import matplotlib.pyplot as plt
import theano
import lasagne
import pickle


class Unpool2DLayer(lasagne.layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(ds[0], axis=2).repeat(ds[1], axis=3)




batch_size = 16
CONTROLDIM=64

'''seeds = np.float32(np.random.rand(batch_size, CONTROLDIM)*2-1)
seedsNew = np.zeros((10,64),dtype=np.float32)
seed1 = np.float32(np.random.rand(1, CONTROLDIM)*2-1)
seed2 = np.float32(np.random.rand(1, CONTROLDIM)*2-1)'''

seed1 = np.squeeze(np.float32(np.random.rand(batch_size / 2, 1)*.3+.7))
seed2 = np.squeeze(np.float32(np.random.rand(batch_size / 2, 1)*.3 - 1))
#seed1 = np.squeeze(np.float32(np.ones([batch_size / 2, 1])))
#seed2 = np.squeeze(np.float32(-np.ones([batch_size / 2, 1])))
seeds1 = np.float32(np.random.rand(batch_size / 2, CONTROLDIM) * 2 - 1)
seeds2 = np.float32(np.random.rand(batch_size / 2, CONTROLDIM) * 2 - 1)
seeds1[:, 0] = seed1
seeds2[:, 0] = seed2
seeds = np.concatenate((seeds1, seeds2), axis=0)

'''seedsDiff = (seed2-seed1)/9
seedsNew[0,:] = seed1
for ii in range(9):
    seedsNew[ii+1,:] = seedsNew[ii,:]+seedsDiff'''
with open('netBestGeneratorCELEBa_BEGAN3singleGenWithClassifierNonBlindAttempt1_2_128x128_Epoch50.pickle','rb') as handle:
    nno = pickle.load(handle)

OUTPUT_image = lasagne.layers.get_output(nno,(seeds))
imageOut = OUTPUT_image.eval()
imageOut = (imageOut - np.min(imageOut))/(np.max(imageOut)-np.min(imageOut))
'''f2, ((axx1, axx2), (axx3, axx4)) = plt.subplots(2, 2, sharey=True)
axx1.imshow(np.squeeze(imageOut[0 , 0 , : , :]), cmap='gray')
axx2.imshow(np.squeeze(imageOut[1 , 0 , : , :]), cmap='gray')
axx3.imshow(np.squeeze(imageOut[2 , 0 , : , :]), cmap='gray')
axx4.imshow(np.squeeze(imageOut[3 , 0 , : , :]), cmap='gray')'''


#plt.imshow(OUTPUT_image[2,0,:,:],cmap='gray')
f2, ((axx1, axx2 , axx3 , axx4), (axx5, axx6, axx7,axx8), (axx9, axx10, axx11,axx12),(axx13, axx14, axx15,axx16)) = plt.subplots(4, 4, sharey=True)
#plt.imshow(imageOut[2,0,:,:],cmap='gray')
axx1.imshow(np.transpose(np.squeeze(imageOut[0 , : , : , :]),axes=[1,2,0]))
axx2.imshow(np.transpose(np.squeeze(imageOut[1 , : , : , :]),axes=[1,2,0]))
axx3.imshow(np.transpose(np.squeeze(imageOut[2 , : , : , :]),axes=[1,2,0]))
axx4.imshow(np.transpose(np.squeeze(imageOut[3 , : , : , :]),axes=[1,2,0]))
axx5.imshow(np.transpose(np.squeeze(imageOut[4 , : , : , :]),axes=[1,2,0]))
axx6.imshow(np.transpose(np.squeeze(imageOut[5 , : , : , :]),axes=[1,2,0]))
axx7.imshow(np.transpose(np.squeeze(imageOut[6 , : , : , :]),axes=[1,2,0]))
axx8.imshow(np.transpose(np.squeeze(imageOut[7 , : , : , :]),axes=[1,2,0]))
axx9.imshow(np.transpose(np.squeeze(imageOut[8 , : , : , :]),axes=[1,2,0]))
axx10.imshow(np.transpose(np.squeeze(imageOut[9 , : , : , :]),axes=[1,2,0]))
axx11.imshow(np.transpose(np.squeeze(imageOut[10 , : , : , :]),axes=[1,2,0]))
axx12.imshow(np.transpose(np.squeeze(imageOut[11 , : , : , :]),axes=[1,2,0]))
axx13.imshow(np.transpose(np.squeeze(imageOut[12 , : , : , :]),axes=[1,2,0]))
axx14.imshow(np.transpose(np.squeeze(imageOut[13 , : , : , :]),axes=[1,2,0]))
axx15.imshow(np.transpose(np.squeeze(imageOut[14 , : , : , :]),axes=[1,2,0]))
axx16.imshow(np.transpose(np.squeeze(imageOut[15 , : , : , :]),axes=[1,2,0]))
plt.show()

print'yup'