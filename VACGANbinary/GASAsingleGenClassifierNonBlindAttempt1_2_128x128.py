import numpy as np
import scipy.io
import theano
import theano.tensor as T
import lasagne
import h5py
import sys
import time
import pickle
from MODELSrgb import GeneratorNetwork128x128, DiscriminatorNetwork128x128, ClassifierNetwork128x128
import matplotlib.pyplot as plt
print 'Loading Data, Please Wait...'
dataClass1train = h5py.File('/home/gwy-dnn/Documents/Project GASA/CELEBa/DB_male_train_128x128.mat')
dataClass1train = dataClass1train['images']
dataClass1train = np.array(dataClass1train)
dataClass1train = np.transpose(dataClass1train , axes=[3,0,2,1])
dataClass1validation = h5py.File('/home/gwy-dnn/Documents/Project GASA/CELEBa/DB_male_validation_128x128.mat')
dataClass1validation = dataClass1validation['images']
dataClass1validation = np.array(dataClass1validation)
dataClass1validation = np.transpose(dataClass1validation , axes=[3,0,2,1])
dataClass1test = h5py.File('/home/gwy-dnn/Documents/Project GASA/CELEBa/DB_male_test_128x128.mat')
dataClass1test = dataClass1test['images']
dataClass1test = np.array(dataClass1test)
dataClass1test = np.transpose(dataClass1test , axes=[3,0,2,1])

dataClass2train = h5py.File('/home/gwy-dnn/Documents/Project GASA/CELEBa/DB_female_train_128x128.mat')
dataClass2train = dataClass2train['images']
dataClass2train = np.array(dataClass2train)
dataClass2train = np.transpose(dataClass2train , axes=[3,0,2,1])
dataClass2validation = h5py.File('/home/gwy-dnn/Documents/Project GASA/CELEBa/DB_female_validation_128x128.mat')
dataClass2validation = dataClass2validation['images']
dataClass2validation = np.array(dataClass2validation)
dataClass2validation = np.transpose(dataClass2validation , axes=[3,0,2,1])
dataClass2test = h5py.File('/home/gwy-dnn/Documents/Project GASA/CELEBa/DB_female_test_128x128.mat')
dataClass2test = dataClass2test['images']
dataClass2test = np.array(dataClass2test)
dataClass2test = np.transpose(dataClass2test , axes=[3,0,2,1])

dataClass1 = np.concatenate((dataClass1train,dataClass1validation,dataClass1test),axis=0)
dataClass2 = np.concatenate((dataClass2train,dataClass2validation,dataClass2test),axis=0)

sys.setrecursionlimit(50000)


num_epochs = 500
#my_loss = 100000
k_t=np.float32(0)
lambda_k = 0.001
gamma = 0.5
alpha = 0.991
beta = 0.009
print 'alpha: '+str(alpha) + ' beta: '+ str(beta)
CONTROLDIM = 64
batch_size = 16
NN = 64

Gen_out_layer = GeneratorNetwork128x128(CONTROLDIM=CONTROLDIM,mini_batch_size=batch_size,NN=NN,name='Gen')
Disc_out_layer = DiscriminatorNetwork128x128(CONTROLDIM=CONTROLDIM,mini_batch_size=batch_size,NN=NN,name='Disc')
networkBOut = ClassifierNetwork128x128(name='Classifier')
print 'BUILDING THE MODEL....'
noise_var = T.matrix(name='noise')
input_var = T.tensor4(name='inputs')
#input_varB = T.tensor4(name='input_B')
target_varB = T.vector(name='target_B')
K_T = T.scalar(name='K_T')

# BEGAN variables
Disc_output_real = lasagne.layers.get_output(Disc_out_layer,inputs=input_var)
Disc_output_fake = lasagne.layers.get_output(Disc_out_layer,
                                              inputs=lasagne.layers.get_output(Gen_out_layer,
                                                                               inputs=noise_var))
Gen_output = lasagne.layers.get_output(Gen_out_layer,inputs=noise_var)

# Classifier variables
NetB_all_input = T.concatenate([lasagne.layers.get_output(Gen_out_layer,inputs=noise_var),
                                input_var], axis=0)

NetB_output_all = T.add(lasagne.layers.get_output(networkBOut,inputs=NetB_all_input),np.finfo(np.float32).eps)

# BEGAN losses
Disc_loss_real = T.abs_(Disc_output_real - input_var)
Disc_loss_real = Disc_loss_real.mean()

Disc_loss_fake = T.abs_(Disc_output_fake - Gen_output)
Disc_loss_fake = Disc_loss_fake.mean()

Gen_loss = T.abs_(Disc_output_fake - Gen_output)
Gen_loss = Gen_loss.mean()

# Classifier losses
#NetB_loss_fake = lasagne.objectives.categorical_crossentropy(NetB_output_fake,targets_fake)
#NetB_loss_fake = NetB_loss_fake.mean()

NetB_loss_all = lasagne.objectives.binary_crossentropy(NetB_output_all,target_varB)
NetB_loss_all = NetB_loss_all.mean()

# Total Losses

Disc_loss_T = Disc_loss_real - K_T * Disc_loss_fake

Gen_loss_T = alpha * Gen_loss + beta * NetB_loss_all

NetB_loss_T = NetB_loss_all

# Parameters

Disc_params = lasagne.layers.get_all_params(Disc_out_layer , trainable=True)

Gen_params = lasagne.layers.get_all_params(Gen_out_layer , trainable=True)

NetB_params = lasagne.layers.get_all_params(networkBOut , trainable=True)

# updates
updates = lasagne.updates.adam(Gen_loss_T,Gen_params,learning_rate=0.0001,beta1=.5,beta2=0.999)
updates.update(lasagne.updates.adam(Disc_loss_T,Disc_params,learning_rate=0.0001,beta1=.5,beta2=0.999))
updates.update(lasagne.updates.nesterov_momentum(NetB_loss_T,NetB_params,learning_rate=0.01,momentum=0.9))

print 'COMPILING THE MODEL... PLEASE WAIT....'
TrainFunction = theano.function([noise_var,input_var,K_T,target_varB],
                                [Disc_loss_T,
                                 Disc_loss_real,
                                 Gen_loss,
                                 NetB_loss_T,
                                 Disc_loss_real + T.abs_(gamma * Disc_loss_real - Gen_loss)],updates=updates)

GEN_LOSS = np.array([])
DISC_LOSS = np.array([])
NETB_LOSS = np.array([])
M_GLOBAL = np.array([])

m_global_value_best = 999999999999999

print 'TRAINING STARTED...'
for epoch in range(num_epochs):
    train_error = 0
    gen_loss_value = 0
    disc_loss_value = 0
    m_global_value = 0
    net_b_loss = 0
    start_time = time.time()
    #carry = np.float32(1)

    for dummy in range(2400):
        seed1 = np.squeeze(np.float32(np.random.rand(batch_size/2,1)))
        seed2 = np.squeeze(np.float32(np.random.rand(batch_size/2,1)-1))
        seeds1 = np.float32(np.random.rand(batch_size/2, CONTROLDIM)*2-1)
        seeds2 = np.float32(np.random.rand(batch_size/2, CONTROLDIM)*2-1)
        seeds1[:,0] = seed1
        seeds2[:,0] = seed2
        seeds = np.concatenate((seeds1,seeds2),axis=0)
        sampleIndices1 = np.random.permutation(dataClass1.shape[0])
        samples1 = dataClass1[sampleIndices1[0:batch_size / 2],]
        sampleIndices2 = np.random.permutation(dataClass2.shape[0])
        samples2 = dataClass2[sampleIndices2[0:batch_size / 2],]
        samples = np.concatenate((samples1,samples2),axis=0)
        samples = np.float32(samples)/np.float32(255)
        #sampleIndices = np.random.permutation(RealDataX.shape[0])
        #samples = RealDataX[sampleIndices[0:batch_size],]
        targets_all_num = np.concatenate([np.zeros([np.int(batch_size / 2), 1]), np.ones([np.int(batch_size / 2), 1])])
        targets_all_num = np.squeeze(np.float32(targets_all_num))
        targets_all_num = np.concatenate([targets_all_num,targets_all_num])
        disc_error, Disc_Loss_real_Num , gen_error,NetB_Loss_Num,M_Global_Num  = TrainFunction(seeds,samples,k_t,targets_all_num)
        k_t = np.float32(np.clip(k_t+lambda_k*(gamma*Disc_Loss_real_Num-gen_error),0,1))
        gen_loss_value += gen_error
        disc_loss_value += disc_error
        m_global_value += M_Global_Num
        net_b_loss += NetB_Loss_Num
        #carry = np.float32(carry- 1.0/1022.0)
        #gen_loss_value += genLossFunction(seeds,samples)
        #disc_loss_value += DiscLossFunction(seeds,samples)
        if dummy%100==0:
            print 'Iteration ' + str(dummy+1) + ' finished successfully.'
    #if m_global_value<m_global_value_best:
        #m_global_value_best=m_global_value
    with open('netBestDiscriminatorCELEBa_BEGAN3singleGenWithClassifierNonBlindAttempt1_2_128x128_Epoch'+str(epoch)+'.pickle', 'wb') as handle:
        print('saving the model Discriminator....')
        pickle.dump(Disc_out_layer, handle)

    with open('netBestGeneratorCELEBa_BEGAN3singleGenWithClassifierNonBlindAttempt1_2_128x128_Epoch'+str(epoch)+'.pickle', 'wb') as handle:
        print('saving the model Generator....')
        pickle.dump(Gen_out_layer, handle)

    '''my_loss_temp = gen_loss_value + disc_loss_value
    if my_loss_temp < my_loss:
        my_loss = my_loss_temp
        with open('netLastDiscriminatorMyLossAERESINJAttemp1BigData.pickle', 'wb') as handle:
            print('saving the model Discriminator MyLoss....')
            pickle.dump(Disc_out_layer, handle)

        with open('netLastGeneratorMyLossAERESINJAttempt1BigData.pickle', 'wb') as handle:
            print('saving the model Generator MyLoss....')
            pickle.dump(Gen_out_layer, handle)'''

    print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    #print("  training loss acc:\t\t{}".format(train_error/10))
    print("  training loss generator:\t\t{:.6f}".format(gen_loss_value / 10))
    print("  training loss discriminator:\t\t{:.6f}".format(disc_loss_value / 10))
    print("  training m_global:\t\t{:.6f}".format(m_global_value))
    print("  training network B Loss:\t\t{:.6f}".format(net_b_loss/2400))
    GEN_LOSS = np.append(GEN_LOSS,gen_loss_value)
    DISC_LOSS = np.append(DISC_LOSS,disc_loss_value)
    M_GLOBAL = np.append(M_GLOBAL,m_global_value)
    NETB_LOSS = np.append(NETB_LOSS,net_b_loss)
    scipy.io.savemat('LossesCELEBa_BEGAN3singleGenWithClassifierNonBlindAttempt1_2_128x128.mat',mdict={'genLoss':GEN_LOSS,'discLoss':DISC_LOSS,'Mglobal':M_GLOBAL,'netBloss':NETB_LOSS})

print'yup'



#dataClass2 = np.float32(Data['imclass2'])

print 'yup'