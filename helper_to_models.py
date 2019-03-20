import keras as K
import numpy as np, json

from keras.utils.np_utils import to_categorical

from keras.engine.topology import get_source_inputs
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.utils import plot_model

from keras.backend import tf as ktf
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, AveragePooling2D
from keras.layers import SeparableConv2D, MaxPooling2D, UpSampling2D, Conv2D, Dropout
from keras.layers import Lambda, Input, BatchNormalization, Concatenate, ZeroPadding2D, Add, Multiply
from keras.models import Model, Sequential, model_from_json
from keras.layers.advanced_activations import PReLU, LeakyReLU
#from keras.applications import ResNet50, VGG19, InceptionV3, MobileNet

from config import *
from helper_custom_layers import *


def resDown(filters, downKernel, input_,):
    down_ = Conv2D(filters, downKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(input_)
    down_ = LeakyReLU(alpha=0.3)(down_)
    

    down_ = Conv2D(filters, downKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(down_)
    down_res = LeakyReLU(alpha=0.3)(down_)
    
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_res)

    return down_pool, down_res


def resUp(filters, upKernel, input_, down_):
    upsample_ = UpSampling2D(size=(2, 2))(input_)

    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(upsample_)
    up_ = LeakyReLU(alpha=0.3)(up_)
    up_ = Concatenate(axis=-1)([down_, up_])

    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(up_)
    up_ = LeakyReLU(alpha=0.3)(up_)

    return up_

def resSumDown(filters, downKernel, input_):
    down_ = Conv2D(filters, downKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(input_)
    down_ = LeakyReLU(alpha=0.3)(down_)

    down_ = Conv2D(filters, downKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(down_)
    down_res = LeakyReLU(alpha=0.3)(down_)

    #apply elemenwise sum operation
    down_res = ElementWiseSum()([down_res, input_])
    
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_res)

    return down_pool, down_res

def UNet1024SkipConnection():
    downKernel = 3;
    upKernel = 3;
    
    inputs = Input(shape=(W,H,1))

    #proceed with unet architecture
    down0, down0_res = resSumDown(FILTER1, downKernel, inputs)
    down1, down1_res = resSumDown(FILTER2, downKernel, down0)
    down2, down2_res = resSumDown(FILTER3, downKernel, down1)
    down3, down3_res = resSumDown(FILTER4, downKernel, down2)
    down4, down4_res = resSumDown(FILTER5, downKernel, down3)
    down5, down5_res = resSumDown(FILTER6, downKernel, down4)

    print(down5.shape)
    center = Conv2D(1024, (1, 1), padding='same',kernel_regularizer=K.regularizers.l2(L2PENALTY))(down5)
    #center = BatchNormalization(epsilon=1e-4)(center)
    center = LeakyReLU(alpha=0.3)(center)
    print(center.shape)

    up5 = resUp(FILTER6, upKernel, center, down5_res)
    up4 = resUp(FILTER5, upKernel, up5, down4_res)
    up3 = resUp(FILTER4, upKernel, up4, down3_res)
    up2 = resUp(FILTER3, upKernel, up3, down2_res)
    up1 = resUp(FILTER2, upKernel, up2, down1_res)
    up0 = resUp(FILTER1, upKernel, up1, down0_res)


    if NUMCLASSES==1:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='sigmoid', name='organ_output', kernel_initializer='glorot_uniform')(up0)
    else:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='softmax', name='organ_output', kernel_initializer='glorot_uniform')(up0)

    model = Model(inputs=inputs, outputs=[organMask],name="model");

    return model




def UNet1024():
    downKernel = 3;
    upKernel = 3;
    
    inputs = Input(shape=(W,H,1))

    #proceed with unet architecture
    down0, down0_res = resDown(FILTER1, downKernel, inputs)
    down1, down1_res = resDown(FILTER2, downKernel, down0)
    down2, down2_res = resDown(FILTER3, downKernel, down1)
    down3, down3_res = resDown(FILTER4, downKernel, down2)
    down4, down4_res = resDown(FILTER5, downKernel, down3)
    down5, down5_res = resDown(FILTER6, downKernel, down4)

    print(down5.shape)
    center = Conv2D(1024, (1, 1), padding='same',kernel_regularizer=K.regularizers.l2(L2PENALTY))(down5)
    #center = BatchNormalization(epsilon=1e-4)(center)
    #center = Activation('relu')(center)
    center = LeakyReLU(alpha=0.3)(center)
    print(center.shape)

    up5 = resUp(FILTER6, upKernel, center, down5_res)
    up4 = resUp(FILTER5, upKernel, up5, down4_res)
    up3 = resUp(FILTER4, upKernel, up4, down3_res)
    up2 = resUp(FILTER3, upKernel, up3, down2_res)
    up1 = resUp(FILTER2, upKernel, up2, down1_res)
    up0 = resUp(FILTER1, upKernel, up1, down0_res)


    if NUMCLASSES==1:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='sigmoid', name='organ_output', kernel_initializer='glorot_uniform')(up0)
    else:
        organMask = Conv2D(NUMCLASSES, (1, 1), activation='softmax', name='organ_output', kernel_initializer='glorot_uniform')(up0)

    model = Model(inputs=inputs, outputs=[organMask],name="model");

    return model


def scaleUp(input_, filters, upKernel, scale=(2,2)):

    up_ = UpSampling2D(size=scale)(input_)
    
    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(up_)
    up_ = LeakyReLU(alpha=0.3)(up_)
    
    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(up_)
    up_ = LeakyReLU(alpha=0.3)(up_)
    
    return up_


def scaleDown(input_, filters, downKernel, scale=(2,2)):

    down_ = Conv2D(filters, downKernel, padding='same', 
                    kernel_initializer='he_uniform', 
                    kernel_regularizer=K.regularizers.l1(L1PENALTY))(input_)
    down_ = LeakyReLU(alpha=0.3)(down_)
    
    down_ = Conv2D(filters, downKernel, padding='same', 
                    kernel_initializer='he_uniform', 
                    kernel_regularizer=K.regularizers.l1(L1PENALTY))(down_)
    down_ = LeakyReLU(alpha=0.3)(down_)

    down_ = MaxPooling2D((2, 2), strides=(2, 2))(down_)

    return down_


def forwardOp(input_, filters, kernel, batch_norm=False):
    
    skip_ = Conv2D(filters, kernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(input_)
    if batch_norm:
        skip_ = BatchNormalization(scale=False, epsilon=1e-4)(skip_)  
    skip_ = LeakyReLU(alpha=0.3)(skip_)
    
    skip_ = Conv2D(filters, kernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(skip_)
    if batch_norm:
        skip_ = BatchNormalization(scale=False, epsilon=1e-4)(skip_)  
    skip_ = LeakyReLU(alpha=0.3)(skip_)

    return skip_
    
def MultiScaleNet():
    downKernel = 3
    upKernel = 3
    scale0_filter = FILTER1 #32
    scale1_filter = FILTER2 #scale0_filter*2
    scale2_filter = FILTER3 #scale0_filter*4
    scale3_filter = FILTER4 #scale0_filter*8
    scale4_filter = FILTER5 #scale0_filter*16
   
    #calculate first diagonal ops
    diag1_scale0 = Input(shape=(W,H,1))
    diag1_scale1 = scaleDown(diag1_scale0, scale1_filter, downKernel, scale=(2,2))
    diag1_scale2 = scaleDown(diag1_scale1, scale2_filter, downKernel, scale=(2,2))
    diag1_scale3 = scaleDown(diag1_scale2, scale3_filter, downKernel, scale=(2,2))
    diag1_scale4 = scaleDown(diag1_scale3, scale4_filter, downKernel, scale=(2,2))

    ##########################
    # Calculate second diagonal
    ########################## 
    # Scale 0
    diag2_scale0 = forwardOp(diag1_scale0, scale0_filter, downKernel, batch_norm = False) #direct path from input
    diag2_scale1_up = scaleUp(diag1_scale1, scale0_filter, upKernel)
    diag2_scale0 = Add()([diag2_scale0, diag2_scale1_up])

    #Scale 1
    diag2_scale1 = forwardOp(diag1_scale1, scale1_filter, downKernel, batch_norm = False)
    diag2_scale2_up = scaleUp(diag1_scale2, scale1_filter, upKernel)
    diag2_scale1 = Add()([diag2_scale1, diag2_scale2_up])

    #Scale 2
    diag2_scale2 = forwardOp(diag1_scale2, scale2_filter, downKernel, batch_norm = False)
    diag2_scale3_up = scaleUp(diag1_scale3, scale2_filter, upKernel)
    diag2_scale2 = Add()([diag2_scale2, diag2_scale3_up])
        
    #Scale 3
    diag2_scale3 = forwardOp(diag1_scale3, scale3_filter, downKernel, batch_norm = False)
    diag2_scale4_up = scaleUp(diag1_scale4, scale3_filter, upKernel)
    diag2_scale3 = Add()([diag2_scale3, diag2_scale4_up])

    ##########################
    # Calculate third diagonal
    ########################## 
    # Scale 0
    diag3_scale0 = forwardOp(diag2_scale0, scale0_filter, downKernel, batch_norm = False) #direct path from input
    diag3_scale1_up = scaleUp(diag2_scale1, scale0_filter, upKernel)
    diag3_scale0 = Add()([diag3_scale0, diag3_scale1_up])

    #Scale 1
    diag3_scale1 = forwardOp(diag2_scale1, scale1_filter, downKernel, batch_norm = False)
    diag3_scale2_up = scaleUp(diag2_scale2, scale1_filter, upKernel)
    diag3_scale1 = Add()([diag3_scale1, diag3_scale2_up])
    
    #Scale 2
    diag3_scale2 = forwardOp(diag2_scale2, scale2_filter, downKernel, batch_norm = False)
    diag3_scale3_up = scaleUp(diag2_scale3, scale2_filter, upKernel)
    diag3_scale2 = Add()([diag3_scale2, diag3_scale3_up])

    ##########################
    # Calculate fourth diagonal
    ########################## 
    # Scale 0
    diag4_scale0 = forwardOp(diag3_scale0, scale0_filter, downKernel, batch_norm = False) #direct path from input
    diag4_scale1_up = scaleUp(diag3_scale1, scale0_filter, upKernel)
    diag4_scale0 = Add()([diag4_scale0, diag4_scale1_up])

    #Scale 1
    diag4_scale1 = forwardOp(diag3_scale1, scale1_filter, downKernel, batch_norm = False)
    diag4_scale2_up = scaleUp(diag3_scale2, scale1_filter, upKernel)
    diag4_scale1 = Add()([diag4_scale1, diag4_scale2_up])

    ##########################
    # Calculate fifth diagonal (output)
    ##########################
    # Scale 0
    diag5_scale0 = forwardOp(diag4_scale0, scale0_filter, downKernel, batch_norm = False) #direct path from input
    diag5_scale1_up = scaleUp(diag4_scale1, scale0_filter, upKernel)
    diag5_scale0 = Concatenate(axis=-1)([diag5_scale0, diag5_scale1_up])
    diag5_scale0 = forwardOp(diag5_scale0, scale1_filter, downKernel, batch_norm = False)

    #concatenate all layers
    organMask = Conv2D(NUMCLASSES, (1, 1), activation='softmax', name='organ_output', kernel_initializer='glorot_uniform')(diag5_scale0)
    model = Model(inputs=diag1_scale0, outputs=[organMask],name="model")

    return model


def batch_relu_conv(input,filters,kernel,scale):
    # net_flow = BatchNormalization(axis=-1)(input)
    net_flow = LeakyReLU(alpha=0.3)(input)
    net_flow = Conv2D(filters, kernel, padding='same', data_format='channels_last', 
                dilation_rate=scale,kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(input)

    return net_flow, input

def DilatedFCN():
    '''
    Adopted from https://arxiv.org/pdf/1707.01992.pdf
    '''
    filters = 16
    kernel = 3

    inputs = Input(shape=(W,H,1))
    #first block
    conv_block = Conv2D(filters, kernel, padding='same', data_format='channels_last',
                    kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(inputs)

    #repeat batch_relu_conv 3 times, with 0 dilation
    #first block
    block1_a, res_block = batch_relu_conv(conv_block,filters,kernel,scale=1)
    block1_a, _ = batch_relu_conv(block1_a,filters,kernel,scale=1)
    block1_a = Add()([block1_a, res_block])
    #second block
    block1_b, res_block = batch_relu_conv(block1_a,filters,kernel,scale=1)
    block1_b, _ = batch_relu_conv(block1_b,filters,kernel,scale=1)
    block1_b = Add()([block1_b, res_block])
    #third block
    block1_c, res_block = batch_relu_conv(block1_b,filters,kernel,scale=1)
    block1_c, _ = batch_relu_conv(block1_c,filters,kernel,scale=1)
    block1_c = Add()([block1_c, res_block])

    #repeat batch_relu_conv 3 times, with 2 dilation
    #first block
    net_flow, res_block = batch_relu_conv(block1_c,filters*2,kernel,scale=2)
    net_flow, _ = batch_relu_conv(net_flow,filters*2,kernel,scale=2)
    net_flow = Concatenate()([net_flow, res_block])
    net_flow = Conv2D(filters*2,(1,1),padding="same",data_format='channels_last', kernel_initializer='he_uniform')(net_flow)
    #second block
    net_flow, res_block = batch_relu_conv(net_flow,filters*2,kernel,scale=2)
    net_flow, _ = batch_relu_conv(net_flow,filters*2,kernel,scale=2)
    net_flow = Add()([net_flow, res_block])
    #third block
    net_flow, res_block = batch_relu_conv(net_flow,filters*2,kernel,scale=2)
    net_flow, _ = batch_relu_conv(net_flow,filters*2,kernel,scale=2)
    net_flow = Add()([net_flow, res_block])

    #repeat batch_relu_conv 3 times, with 4 dilation
    #first block
    net_flow, res_block = batch_relu_conv(net_flow,filters*4,kernel,scale=4)
    net_flow, _ = batch_relu_conv(net_flow,filters*4,kernel,scale=4)
    net_flow = Concatenate()([net_flow, res_block])
    net_flow = Conv2D(filters*4,(1,1),padding="same",data_format='channels_last', kernel_initializer='he_uniform')(net_flow)
    #second block
    net_flow, res_block = batch_relu_conv(net_flow,filters*4,kernel,scale=4)
    net_flow, _ = batch_relu_conv(net_flow,filters*4,kernel,scale=4)
    net_flow = Add()([net_flow, res_block])
    #third block
    net_flow, res_block = batch_relu_conv(net_flow,filters*4,kernel,scale=4)
    net_flow, _ = batch_relu_conv(net_flow,filters*4,kernel,scale=4)
    net_flow = Add()([net_flow, res_block])

    #final layer
    organMask = Conv2D(NUMCLASSES, (1, 1), activation='softmax', data_format='channels_last', 
                    name='organ_output', kernel_initializer='glorot_uniform')(net_flow)
    model = Model(inputs=inputs, outputs=[organMask],name="model")

    return model
