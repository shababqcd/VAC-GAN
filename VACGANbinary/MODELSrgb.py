import lasagne
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



def GeneratorNetwork48x48(CONTROLDIM = 64,mini_batch_size = 16,NN = 64,name = 'Gen'):

    GenLinput = lasagne.layers.InputLayer(shape=(None, CONTROLDIM), name=name+'Linput')
    GenL1 = lasagne.layers.DenseLayer(incoming=GenLinput, num_units=6 * 6 * NN,
                                       nonlinearity=lasagne.nonlinearities.identity,
                                       name=name+'L1')
    GenL1Reshape = lasagne.layers.ReshapeLayer(incoming=GenL1, shape=(mini_batch_size, NN, 6, 6),
                                                name=name+'L1Reshape')
    GenL2 = lasagne.layers.Conv2DLayer(GenL1Reshape, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L2')
    GenL3 = lasagne.layers.Conv2DLayer(GenL2, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L3')
    GenL3UP = Unpool2DLayer(GenL3, ds=(2, 2), name=name+'L3UP')
    GenL4 = lasagne.layers.Conv2DLayer(GenL3UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L4')
    GenL5 = lasagne.layers.Conv2DLayer(GenL4, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L5')
    GenL5UP = Unpool2DLayer(GenL5, ds=(2, 2), name=name+'L5UP')
    GenL6 = lasagne.layers.Conv2DLayer(GenL5UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L6')
    GenL7 = lasagne.layers.Conv2DLayer(GenL6, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L7')
    GenL7UP = Unpool2DLayer(GenL7, ds=(2, 2), name=name+'L7UP')
    GenL8 = lasagne.layers.Conv2DLayer(GenL7UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L8')
    GenL9 = lasagne.layers.Conv2DLayer(GenL8, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L9')
    Gen_out_layer = lasagne.layers.Conv2DLayer(incoming=GenL9, num_filters=3, filter_size=(3, 3),
                                                pad='same', nonlinearity=lasagne.nonlinearities.identity,name=name+'_out_layer')
    return Gen_out_layer

def DiscriminatorNetwork48x48(CONTROLDIM = 64,mini_batch_size = 16,NN = 64,name = 'Disc'):

    DiscLinput = lasagne.layers.InputLayer(shape=(None, 3, 48, 48), name=name+'Linput')
    DiscL1 = lasagne.layers.Conv2DLayer(DiscLinput, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L1')
    DiscL2 = lasagne.layers.Conv2DLayer(DiscL1, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L2')
    DiscL3 = lasagne.layers.Conv2DLayer(DiscL2, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L3')
    DiscL4 = lasagne.layers.Conv2DLayer(DiscL3, num_filters=2 * NN, filter_size=(1, 1), pad='same',
                                         nonlinearity=lasagne.nonlinearities.identity, name=name+'L4')
    DiscL4MP = lasagne.layers.Pool2DLayer(DiscL4, pool_size=(2, 2), mode='average_inc_pad', name=name+'L4MP')
    DiscL5 = lasagne.layers.Conv2DLayer(DiscL4MP, num_filters=2 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L5')
    DiscL6 = lasagne.layers.Conv2DLayer(DiscL5, num_filters=2 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L6')
    DiscL7 = lasagne.layers.Conv2DLayer(DiscL6, num_filters=3 * NN, filter_size=(1, 1), pad='same',
                                         nonlinearity=lasagne.nonlinearities.identity, name=name+'L7')
    DiscL7MP = lasagne.layers.Pool2DLayer(DiscL7, pool_size=(2, 2), mode='average_inc_pad', name=name+'L7MP')
    DiscL8 = lasagne.layers.Conv2DLayer(DiscL7MP, num_filters=3 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L8')
    DiscL9 = lasagne.layers.Conv2DLayer(DiscL8, num_filters=3 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L9')

    DiscL10 = lasagne.layers.Conv2DLayer(DiscL9, num_filters=4 * NN, filter_size=(1, 1), pad='same',
                                          nonlinearity=lasagne.nonlinearities.identity, name=name+'L10')
    DiscL10MP = lasagne.layers.Pool2DLayer(DiscL10, pool_size=(2, 2), mode='average_inc_pad', name=name+'L10MP')
    DiscL11 = lasagne.layers.Conv2DLayer(DiscL10MP, num_filters=4 * NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L11')
    DiscL12 = lasagne.layers.Conv2DLayer(DiscL11, num_filters=4 * NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L12')
    DiscL13FC = lasagne.layers.DenseLayer(DiscL12, num_units=CONTROLDIM, nonlinearity=lasagne.nonlinearities.identity,
                                           name=name+'L13FC')
    DiscL14 = lasagne.layers.DenseLayer(DiscL13FC, num_units=6 * 6 * NN, nonlinearity=lasagne.nonlinearities.identity,
                                         name=name+'L14')
    DiscL14Reshape = lasagne.layers.ReshapeLayer(incoming=DiscL14, shape=(mini_batch_size, NN, 6, 6),
                                                  name=name+'L14Reshape')
    DiscL15 = lasagne.layers.Conv2DLayer(DiscL14Reshape, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L15')
    DiscL16 = lasagne.layers.Conv2DLayer(DiscL15, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L16')
    DiscL16UP = Unpool2DLayer(DiscL16, ds=(2, 2), name=name+'L16UP')
    DiscL17 = lasagne.layers.Conv2DLayer(DiscL16UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L17')
    DiscL18 = lasagne.layers.Conv2DLayer(DiscL17, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L18')
    DiscL18UP = Unpool2DLayer(DiscL18, ds=(2, 2), name=name+'L18UP')
    DiscL19 = lasagne.layers.Conv2DLayer(DiscL18UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L19')
    DiscL20 = lasagne.layers.Conv2DLayer(DiscL19, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L20')
    DiscL20UP = Unpool2DLayer(DiscL20, ds=(2, 2), name=name+'L20UP')
    DiscL21 = lasagne.layers.Conv2DLayer(DiscL20UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L21')
    DiscL22 = lasagne.layers.Conv2DLayer(DiscL21, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L22')

    Disc_out_layer = lasagne.layers.Conv2DLayer(incoming=DiscL22, num_filters=3, filter_size=(3, 3),
                                                 pad='same', nonlinearity=lasagne.nonlinearities.identity,
                                                 name=name+'_out_layer')
    return Disc_out_layer


def ClassifierNetwork48x48(name='Classifier'):
    networkBi = lasagne.layers.InputLayer(shape=(None, 3, 48, 48), name=name+'Linput')
    netBL1 = lasagne.layers.Conv2DLayer(networkBi, num_filters=16, filter_size=(3, 3), name=name+'L1')
    netBL1MP = lasagne.layers.MaxPool2DLayer(netBL1, pool_size=(2, 2), name=name+'L1MP')
    netBL2 = lasagne.layers.Conv2DLayer(netBL1MP, num_filters=8, filter_size=(3, 3), name=name+'L2')
    netBL2MP = lasagne.layers.MaxPool2DLayer(netBL2, pool_size=(2, 2), name=name+'L2MP')
    netBL3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL2MP), num_units=1024, name=name+'L3')
    networkBOut = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL3), num_units=1,
                                            nonlinearity=lasagne.nonlinearities.sigmoid, name=name+'_out_layer')
    return networkBOut



def GeneratorNetwork64x64(CONTROLDIM = 64,mini_batch_size = 16,NN = 64,name = 'Gen'):

    GenLinput = lasagne.layers.InputLayer(shape=(None, CONTROLDIM), name=name+'Linput')
    GenL1 = lasagne.layers.DenseLayer(incoming=GenLinput, num_units=8 * 8 * NN,
                                       nonlinearity=lasagne.nonlinearities.identity,
                                       name=name+'L1')
    GenL1Reshape = lasagne.layers.ReshapeLayer(incoming=GenL1, shape=(mini_batch_size, NN, 8, 8),
                                                name=name+'L1Reshape')
    GenL2 = lasagne.layers.Conv2DLayer(GenL1Reshape, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L2')
    GenL3 = lasagne.layers.Conv2DLayer(GenL2, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L3')
    GenL3UP = Unpool2DLayer(GenL3, ds=(2, 2), name=name+'L3UP')
    GenL4 = lasagne.layers.Conv2DLayer(GenL3UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L4')
    GenL5 = lasagne.layers.Conv2DLayer(GenL4, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L5')
    GenL5UP = Unpool2DLayer(GenL5, ds=(2, 2), name=name+'L5UP')
    GenL6 = lasagne.layers.Conv2DLayer(GenL5UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L6')
    GenL7 = lasagne.layers.Conv2DLayer(GenL6, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L7')
    GenL7UP = Unpool2DLayer(GenL7, ds=(2, 2), name=name+'L7UP')
    GenL8 = lasagne.layers.Conv2DLayer(GenL7UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L8')
    GenL9 = lasagne.layers.Conv2DLayer(GenL8, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L9')
    Gen_out_layer = lasagne.layers.Conv2DLayer(incoming=GenL9, num_filters=3, filter_size=(3, 3),
                                                pad='same', nonlinearity=lasagne.nonlinearities.identity,name=name+'_out_layer')
    return Gen_out_layer

def DiscriminatorNetwork64x64(CONTROLDIM = 64,mini_batch_size = 16,NN = 64,name = 'Disc'):

    DiscLinput = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), name=name+'Linput')
    DiscL1 = lasagne.layers.Conv2DLayer(DiscLinput, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L1')
    DiscL2 = lasagne.layers.Conv2DLayer(DiscL1, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L2')
    DiscL3 = lasagne.layers.Conv2DLayer(DiscL2, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L3')
    DiscL4 = lasagne.layers.Conv2DLayer(DiscL3, num_filters=2 * NN, filter_size=(1, 1), pad='same',
                                         nonlinearity=lasagne.nonlinearities.identity, name=name+'L4')
    DiscL4MP = lasagne.layers.Pool2DLayer(DiscL4, pool_size=(2, 2), mode='average_inc_pad', name=name+'L4MP')
    DiscL5 = lasagne.layers.Conv2DLayer(DiscL4MP, num_filters=2 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L5')
    DiscL6 = lasagne.layers.Conv2DLayer(DiscL5, num_filters=2 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L6')
    DiscL7 = lasagne.layers.Conv2DLayer(DiscL6, num_filters=3 * NN, filter_size=(1, 1), pad='same',
                                         nonlinearity=lasagne.nonlinearities.identity, name=name+'L7')
    DiscL7MP = lasagne.layers.Pool2DLayer(DiscL7, pool_size=(2, 2), mode='average_inc_pad', name=name+'L7MP')
    DiscL8 = lasagne.layers.Conv2DLayer(DiscL7MP, num_filters=3 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L8')
    DiscL9 = lasagne.layers.Conv2DLayer(DiscL8, num_filters=3 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name+'L9')

    DiscL10 = lasagne.layers.Conv2DLayer(DiscL9, num_filters=4 * NN, filter_size=(1, 1), pad='same',
                                          nonlinearity=lasagne.nonlinearities.identity, name=name+'L10')
    DiscL10MP = lasagne.layers.Pool2DLayer(DiscL10, pool_size=(2, 2), mode='average_inc_pad', name=name+'L10MP')
    DiscL11 = lasagne.layers.Conv2DLayer(DiscL10MP, num_filters=4 * NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L11')
    DiscL12 = lasagne.layers.Conv2DLayer(DiscL11, num_filters=4 * NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L12')
    DiscL13FC = lasagne.layers.DenseLayer(DiscL12, num_units=CONTROLDIM, nonlinearity=lasagne.nonlinearities.identity,
                                           name=name+'L13FC')
    DiscL14 = lasagne.layers.DenseLayer(DiscL13FC, num_units=8 * 8 * NN, nonlinearity=lasagne.nonlinearities.identity,
                                         name=name+'L14')
    DiscL14Reshape = lasagne.layers.ReshapeLayer(incoming=DiscL14, shape=(mini_batch_size, NN, 8, 8),
                                                  name=name+'L14Reshape')
    DiscL15 = lasagne.layers.Conv2DLayer(DiscL14Reshape, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L15')
    DiscL16 = lasagne.layers.Conv2DLayer(DiscL15, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L16')
    DiscL16UP = Unpool2DLayer(DiscL16, ds=(2, 2), name=name+'L16UP')
    DiscL17 = lasagne.layers.Conv2DLayer(DiscL16UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L17')
    DiscL18 = lasagne.layers.Conv2DLayer(DiscL17, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L18')
    DiscL18UP = Unpool2DLayer(DiscL18, ds=(2, 2), name=name+'L18UP')
    DiscL19 = lasagne.layers.Conv2DLayer(DiscL18UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L19')
    DiscL20 = lasagne.layers.Conv2DLayer(DiscL19, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L20')
    DiscL20UP = Unpool2DLayer(DiscL20, ds=(2, 2), name=name+'L20UP')
    DiscL21 = lasagne.layers.Conv2DLayer(DiscL20UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L21')
    DiscL22 = lasagne.layers.Conv2DLayer(DiscL21, num_filters=NN, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.elu, name=name+'L22')

    Disc_out_layer = lasagne.layers.Conv2DLayer(incoming=DiscL22, num_filters=3, filter_size=(3, 3),
                                                 pad='same', nonlinearity=lasagne.nonlinearities.identity,
                                                 name=name+'_out_layer')
    return Disc_out_layer

def ClassifierNetwork64x64(name='Classifier'):
    networkBi = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), name=name+'Linput')
    netBL1 = lasagne.layers.Conv2DLayer(networkBi, num_filters=16, filter_size=(3, 3), name=name+'L1')
    netBL1MP = lasagne.layers.MaxPool2DLayer(netBL1, pool_size=(2, 2), name=name+'L1MP')
    netBL2 = lasagne.layers.Conv2DLayer(netBL1MP, num_filters=8, filter_size=(3, 3), name=name+'L2')
    netBL2MP = lasagne.layers.MaxPool2DLayer(netBL2, pool_size=(2, 2), name=name+'L2MP')
    netBL3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL2MP), num_units=1024, name=name+'L3')
    networkBOut = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL3), num_units=1,
                                            nonlinearity=lasagne.nonlinearities.sigmoid, name=name+'_out_layer')
    return networkBOut


def GeneratorNetwork96x96(CONTROLDIM = 64,mini_batch_size = 16,NN = 64,name = 'Gen'):
    GenLinput = lasagne.layers.InputLayer(shape=(None,CONTROLDIM),name=name+'Linput')
    GenL1 = lasagne.layers.DenseLayer(incoming=GenLinput, num_units=6 * 6 * NN,
                                       nonlinearity=lasagne.nonlinearities.identity,
                                       name=name+'L1')
    GenL1Reshape = lasagne.layers.ReshapeLayer(incoming=GenL1, shape=(mini_batch_size, NN, 6, 6),
                                                name=name+'L1Reshape')
    GenL2 = lasagne.layers.Conv2DLayer(GenL1Reshape, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L2')
    GenL3 = lasagne.layers.Conv2DLayer(GenL2, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L3')
    GenL3UP = Unpool2DLayer(GenL3, ds=(2, 2), name=name+'L3UP')
    GenL4 = lasagne.layers.Conv2DLayer(GenL3UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L4')
    GenL5 = lasagne.layers.Conv2DLayer(GenL4, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L5')
    GenL5UP = Unpool2DLayer(GenL5, ds=(2, 2), name=name+'L5UP')
    GenL6 = lasagne.layers.Conv2DLayer(GenL5UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L6')
    GenL7 = lasagne.layers.Conv2DLayer(GenL6, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L7')
    GenL7UP = Unpool2DLayer(GenL7, ds=(2, 2), name=name+'L7UP')
    GenL8 = lasagne.layers.Conv2DLayer(GenL7UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L8')
    GenL9 = lasagne.layers.Conv2DLayer(GenL8, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L9')
    GenL9UP = Unpool2DLayer(GenL9, ds=(2, 2), name=name+'L9UP')
    GenL10 = lasagne.layers.Conv2DLayer(GenL9UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L10')
    GenL11 = lasagne.layers.Conv2DLayer(GenL10, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L11')
    Gen_out_layer = lasagne.layers.Conv2DLayer(incoming=GenL11, num_filters=3, filter_size=(3, 3),
                                                pad='same', nonlinearity=lasagne.nonlinearities.identity,name=name+'_out_layer')
    return Gen_out_layer


def DiscriminatorNetwork96x96(CONTROLDIM = 64,mini_batch_size = 16,NN = 64,name = 'Disc'):
    DiscLinput = lasagne.layers.InputLayer(shape=(None, 3, 96, 96), name=name + 'Linput')
    DiscL1 = lasagne.layers.Conv2DLayer(DiscLinput, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L1')
    DiscL2 = lasagne.layers.Conv2DLayer(DiscL1, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L2')
    DiscL3 = lasagne.layers.Conv2DLayer(DiscL2, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L3')
    DiscL4 = lasagne.layers.Conv2DLayer(DiscL3, num_filters=2 * NN, filter_size=(1, 1), pad='same',
                                        nonlinearity=lasagne.nonlinearities.identity, name=name + 'L4')
    DiscL4MP = lasagne.layers.Pool2DLayer(DiscL4, pool_size=(2, 2), mode='average_inc_pad', name=name + 'L4MP')
    DiscL5 = lasagne.layers.Conv2DLayer(DiscL4MP, num_filters=2 * NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L5')
    DiscL6 = lasagne.layers.Conv2DLayer(DiscL5, num_filters=2 * NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L6')
    DiscL7 = lasagne.layers.Conv2DLayer(DiscL6, num_filters=3 * NN, filter_size=(1, 1), pad='same',
                                        nonlinearity=lasagne.nonlinearities.identity, name=name + 'L7')
    DiscL7MP = lasagne.layers.Pool2DLayer(DiscL7, pool_size=(2, 2), mode='average_inc_pad', name=name + 'L7MP')
    DiscL8 = lasagne.layers.Conv2DLayer(DiscL7MP, num_filters=3 * NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L8')
    DiscL9 = lasagne.layers.Conv2DLayer(DiscL8, num_filters=3 * NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L9')

    DiscL10 = lasagne.layers.Conv2DLayer(DiscL9, num_filters=4 * NN, filter_size=(1, 1), pad='same',
                                         nonlinearity=lasagne.nonlinearities.identity, name=name + 'L10')
    DiscL10MP = lasagne.layers.Pool2DLayer(DiscL10, pool_size=(2, 2), mode='average_inc_pad', name=name + 'L10MP')
    DiscL11 = lasagne.layers.Conv2DLayer(DiscL10MP, num_filters=4 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L11')
    DiscL12 = lasagne.layers.Conv2DLayer(DiscL11, num_filters=4 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L12')
    DiscL13 = lasagne.layers.Conv2DLayer(DiscL12, num_filters=5 * NN, filter_size=(1, 1), pad='same',
                                         nonlinearity=lasagne.nonlinearities.identity, name=name + 'L13')
    DiscL13MP = lasagne.layers.Pool2DLayer(DiscL13, pool_size=(2, 2), mode='average_inc_pad', name=name + 'L13MP')
    DiscL14 = lasagne.layers.Conv2DLayer(DiscL13MP, num_filters=5 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L14')
    DiscL15 = lasagne.layers.Conv2DLayer(DiscL14, num_filters=5 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L15')
    DiscL16FC = lasagne.layers.DenseLayer(DiscL15, num_units=CONTROLDIM, nonlinearity=lasagne.nonlinearities.identity,
                                          name=name + 'L16FC')
    DiscL17 = lasagne.layers.DenseLayer(DiscL16FC, num_units=6 * 6 * NN, nonlinearity=lasagne.nonlinearities.identity,
                                        name=name + 'L17')
    DiscL17Reshape = lasagne.layers.ReshapeLayer(incoming=DiscL17, shape=(mini_batch_size, NN, 6, 6),
                                                 name=name + 'L17Reshape')
    DiscL18 = lasagne.layers.Conv2DLayer(DiscL17Reshape, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L18')
    DiscL19 = lasagne.layers.Conv2DLayer(DiscL18, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L19')
    DiscL19UP = Unpool2DLayer(DiscL19, ds=(2, 2), name=name + 'L19UP')
    DiscL20 = lasagne.layers.Conv2DLayer(DiscL19UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L20')
    DiscL21 = lasagne.layers.Conv2DLayer(DiscL20, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L21')
    DiscL21UP = Unpool2DLayer(DiscL21, ds=(2, 2), name=name + 'L21UP')
    DiscL22 = lasagne.layers.Conv2DLayer(DiscL21UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L22')
    DiscL23 = lasagne.layers.Conv2DLayer(DiscL22, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L23')
    DiscL23UP = Unpool2DLayer(DiscL23, ds=(2, 2), name=name + 'L23UP')
    DiscL24 = lasagne.layers.Conv2DLayer(DiscL23UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L24')
    DiscL25 = lasagne.layers.Conv2DLayer(DiscL24, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L25')
    DiscL25UP = Unpool2DLayer(DiscL25, ds=(2, 2), name=name + 'L25UP')
    DiscL26 = lasagne.layers.Conv2DLayer(DiscL25UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L26')
    DiscL27 = lasagne.layers.Conv2DLayer(DiscL26, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L27')

    Disc_out_layer = lasagne.layers.Conv2DLayer(incoming=DiscL27, num_filters=3, filter_size=(3, 3),
                                                pad='same', nonlinearity=lasagne.nonlinearities.identity,
                                                name=name + '_out_layer')
    return Disc_out_layer

def ClassifierNetwork96x96(name='Classifier'):
    networkBi = lasagne.layers.InputLayer(shape=(None, 3, 96, 96), name=name+'Linput')
    netBL1 = lasagne.layers.Conv2DLayer(networkBi, num_filters=16, filter_size=(3, 3), name=name+'L1')
    netBL1MP = lasagne.layers.MaxPool2DLayer(netBL1, pool_size=(3, 3), name=name+'L1MP')
    netBL2 = lasagne.layers.Conv2DLayer(netBL1MP, num_filters=8, filter_size=(3, 3), name=name+'L2')
    netBL2MP = lasagne.layers.MaxPool2DLayer(netBL2, pool_size=(3, 3), name=name+'L2MP')
    netBL3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL2MP), num_units=1024, name=name+'L3')
    networkBOut = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL3), num_units=1,
                                            nonlinearity=lasagne.nonlinearities.sigmoid, name=name+'_out_layer')
    return networkBOut


def GeneratorNetwork128x128(CONTROLDIM = 64,mini_batch_size = 16,NN = 64,name = 'Gen'):
    GenLinput = lasagne.layers.InputLayer(shape=(None,CONTROLDIM),name=name+'Linput')
    GenL1 = lasagne.layers.DenseLayer(incoming=GenLinput, num_units=8 * 8 * NN,
                                       nonlinearity=lasagne.nonlinearities.identity,
                                       name=name+'L1')
    GenL1Reshape = lasagne.layers.ReshapeLayer(incoming=GenL1, shape=(mini_batch_size, NN, 8, 8),
                                                name=name+'L1Reshape')
    GenL2 = lasagne.layers.Conv2DLayer(GenL1Reshape, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L2')
    GenL3 = lasagne.layers.Conv2DLayer(GenL2, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L3')
    GenL3UP = Unpool2DLayer(GenL3, ds=(2, 2), name=name+'L3UP')
    GenL4 = lasagne.layers.Conv2DLayer(GenL3UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L4')
    GenL5 = lasagne.layers.Conv2DLayer(GenL4, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L5')
    GenL5UP = Unpool2DLayer(GenL5, ds=(2, 2), name=name+'L5UP')
    GenL6 = lasagne.layers.Conv2DLayer(GenL5UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L6')
    GenL7 = lasagne.layers.Conv2DLayer(GenL6, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L7')
    GenL7UP = Unpool2DLayer(GenL7, ds=(2, 2), name=name+'L7UP')
    GenL8 = lasagne.layers.Conv2DLayer(GenL7UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L8')
    GenL9 = lasagne.layers.Conv2DLayer(GenL8, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L9')
    GenL9UP = Unpool2DLayer(GenL9, ds=(2, 2), name=name+'L9UP')
    GenL10 = lasagne.layers.Conv2DLayer(GenL9UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L10')
    GenL11 = lasagne.layers.Conv2DLayer(GenL10, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name+'L11')
    Gen_out_layer = lasagne.layers.Conv2DLayer(incoming=GenL11, num_filters=3, filter_size=(3, 3),
                                                pad='same', nonlinearity=lasagne.nonlinearities.identity,name=name+'_out_layer')
    return Gen_out_layer


def DiscriminatorNetwork128x128(CONTROLDIM = 64,mini_batch_size = 16,NN = 64,name = 'Disc'):
    DiscLinput = lasagne.layers.InputLayer(shape=(None, 3, 128, 128), name=name + 'Linput')
    DiscL1 = lasagne.layers.Conv2DLayer(DiscLinput, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L1')
    DiscL2 = lasagne.layers.Conv2DLayer(DiscL1, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L2')
    DiscL3 = lasagne.layers.Conv2DLayer(DiscL2, num_filters=NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L3')
    DiscL4 = lasagne.layers.Conv2DLayer(DiscL3, num_filters=2 * NN, filter_size=(1, 1), pad='same',
                                        nonlinearity=lasagne.nonlinearities.identity, name=name + 'L4')
    DiscL4MP = lasagne.layers.Pool2DLayer(DiscL4, pool_size=(2, 2), mode='average_inc_pad', name=name + 'L4MP')
    DiscL5 = lasagne.layers.Conv2DLayer(DiscL4MP, num_filters=2 * NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L5')
    DiscL6 = lasagne.layers.Conv2DLayer(DiscL5, num_filters=2 * NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L6')
    DiscL7 = lasagne.layers.Conv2DLayer(DiscL6, num_filters=3 * NN, filter_size=(1, 1), pad='same',
                                        nonlinearity=lasagne.nonlinearities.identity, name=name + 'L7')
    DiscL7MP = lasagne.layers.Pool2DLayer(DiscL7, pool_size=(2, 2), mode='average_inc_pad', name=name + 'L7MP')
    DiscL8 = lasagne.layers.Conv2DLayer(DiscL7MP, num_filters=3 * NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L8')
    DiscL9 = lasagne.layers.Conv2DLayer(DiscL8, num_filters=3 * NN, filter_size=(3, 3), pad='same',
                                        nonlinearity=lasagne.nonlinearities.elu, name=name + 'L9')

    DiscL10 = lasagne.layers.Conv2DLayer(DiscL9, num_filters=4 * NN, filter_size=(1, 1), pad='same',
                                         nonlinearity=lasagne.nonlinearities.identity, name=name + 'L10')
    DiscL10MP = lasagne.layers.Pool2DLayer(DiscL10, pool_size=(2, 2), mode='average_inc_pad', name=name + 'L10MP')
    DiscL11 = lasagne.layers.Conv2DLayer(DiscL10MP, num_filters=4 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L11')
    DiscL12 = lasagne.layers.Conv2DLayer(DiscL11, num_filters=4 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L12')
    DiscL13 = lasagne.layers.Conv2DLayer(DiscL12, num_filters=5 * NN, filter_size=(1, 1), pad='same',
                                         nonlinearity=lasagne.nonlinearities.identity, name=name + 'L13')
    DiscL13MP = lasagne.layers.Pool2DLayer(DiscL13, pool_size=(2, 2), mode='average_inc_pad', name=name + 'L13MP')
    DiscL14 = lasagne.layers.Conv2DLayer(DiscL13MP, num_filters=5 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L14')
    DiscL15 = lasagne.layers.Conv2DLayer(DiscL14, num_filters=5 * NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L15')
    DiscL16FC = lasagne.layers.DenseLayer(DiscL15, num_units=CONTROLDIM, nonlinearity=lasagne.nonlinearities.identity,
                                          name=name + 'L16FC')
    DiscL17 = lasagne.layers.DenseLayer(DiscL16FC, num_units=8 * 8 * NN, nonlinearity=lasagne.nonlinearities.identity,
                                        name=name + 'L17')
    DiscL17Reshape = lasagne.layers.ReshapeLayer(incoming=DiscL17, shape=(mini_batch_size, NN, 8, 8),
                                                 name=name + 'L17Reshape')
    DiscL18 = lasagne.layers.Conv2DLayer(DiscL17Reshape, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L18')
    DiscL19 = lasagne.layers.Conv2DLayer(DiscL18, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L19')
    DiscL19UP = Unpool2DLayer(DiscL19, ds=(2, 2), name=name + 'L19UP')
    DiscL20 = lasagne.layers.Conv2DLayer(DiscL19UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L20')
    DiscL21 = lasagne.layers.Conv2DLayer(DiscL20, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L21')
    DiscL21UP = Unpool2DLayer(DiscL21, ds=(2, 2), name=name + 'L21UP')
    DiscL22 = lasagne.layers.Conv2DLayer(DiscL21UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L22')
    DiscL23 = lasagne.layers.Conv2DLayer(DiscL22, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L23')
    DiscL23UP = Unpool2DLayer(DiscL23, ds=(2, 2), name=name + 'L23UP')
    DiscL24 = lasagne.layers.Conv2DLayer(DiscL23UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L24')
    DiscL25 = lasagne.layers.Conv2DLayer(DiscL24, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L25')
    DiscL25UP = Unpool2DLayer(DiscL25, ds=(2, 2), name=name + 'L25UP')
    DiscL26 = lasagne.layers.Conv2DLayer(DiscL25UP, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L26')
    DiscL27 = lasagne.layers.Conv2DLayer(DiscL26, num_filters=NN, filter_size=(3, 3), pad='same',
                                         nonlinearity=lasagne.nonlinearities.elu, name=name + 'L27')

    Disc_out_layer = lasagne.layers.Conv2DLayer(incoming=DiscL27, num_filters=3, filter_size=(3, 3),
                                                pad='same', nonlinearity=lasagne.nonlinearities.identity,
                                                name=name + '_out_layer')
    return Disc_out_layer

def ClassifierNetwork128x128(name='Classifier'):
    networkBi = lasagne.layers.InputLayer(shape=(None, 3, 128, 128), name=name+'Linput')
    netBL1 = lasagne.layers.Conv2DLayer(networkBi, num_filters=16, filter_size=(3, 3), name=name+'L1')
    netBL1MP = lasagne.layers.MaxPool2DLayer(netBL1, pool_size=(3, 3), name=name+'L1MP')
    netBL2 = lasagne.layers.Conv2DLayer(netBL1MP, num_filters=8, filter_size=(3, 3), name=name+'L2')
    netBL2MP = lasagne.layers.MaxPool2DLayer(netBL2, pool_size=(3, 3), name=name+'L2MP')
    netBL3 = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL2MP), num_units=1024, name=name+'L3')
    networkBOut = lasagne.layers.DenseLayer(lasagne.layers.dropout(netBL3), num_units=1,
                                            nonlinearity=lasagne.nonlinearities.sigmoid, name=name+'_out_layer')
    return networkBOut


