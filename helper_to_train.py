import numpy as np
import json
from helper_to_models import *
from helper_custom_layers import ElementWiseSum
from config import *

import tensorflow as tf
import keras as K
from keras.models import load_model, model_from_json
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    #fix y_pred
    _epsilon = tf.convert_to_tensor(K.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    # skip the class axis for calculating Dice score
    numerator = 2. * K.backend.sum(y_pred * y_true, axis=(0,1,2))
    denominator = K.backend.sum(K.backend.square(y_pred) + K.backend.square(y_true),axis=(0,1,2))

    # average over classes and batch
    return 1 - K.backend.mean(numerator / (denominator + epsilon))
    

def dist_loss(y_true,y_pred):
    '''
    Method to calculate absoltute distance contours. using L1 norm
    '''
    _epsilon = tf.convert_to_tensor(K.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    y_true = tf.clip_by_value(y_true, _epsilon, 1. - _epsilon)

    #first convert predection to categorical labels
    #y_true = tf.Print(y_true, [tf.argmax(y_true,axis=-1)[:,128,186]], message="True: ", summarize=8)
    #y_labels = tf.sign(y_pred - tf.reduce_max(y_pred,axis=-1,keep_dims=True))
    #y_labels = tf.add(y_labels, tf.ones_like(y_labels))
    #y_labels = tf.Print(y_labels, [tf.argmax(y_labels,axis=-1)[:,128,186]], message="Pred: ", summarize=8)

    #now calculate difference per batch per class
    diff = tf.reduce_sum(tf.abs(y_true - y_pred),axis=(1,2))
    # diff = tf.Print(diff, [diff], message="Diff: ", summarize=8)
    #normalize differences by the frequencis
    true_norm = tf.reduce_sum(y_true,axis=(1,2))
    # true_norm = tf.Print(true_norm, [true_norm], message="Freq: ", summarize=8)
    pred_norm = tf.reduce_sum(y_pred,axis=(1,2))

    #if class is absent assing 1 so, no zero division
    w_mask = tf.equal(tf.cast(true_norm,tf.int64),0)
    norm = tf.where(w_mask,y=true_norm,x=tf.ones_like(true_norm))
    
    #divide differences by norm
    diff_norm = tf.reduce_sum(tf.div(diff,norm+pred_norm),axis=0)
    # diff_norm = tf.Print(diff_norm, [diff_norm], message="Norm: ", summarize=8)

    #calculate mean distances
    return tf.reduce_mean(diff_norm)
    

def tversky_score(y_true, y_pred, alpha = 0.5, beta = 0.5):

    true_positives = y_true * y_pred;
    false_negatives = y_true * (1 - y_pred);
    false_positives = (1 - y_true) * y_pred;

    num = K.backend.sum(true_positives, axis = (0,1,2)) #compute loss per-batch
    den = num+alpha*K.backend.sum(false_negatives, axis = (0,1,2)) + beta*K.backend.sum(false_positives, axis=(0,1,2))+1
    T = K.backend.mean(num/den)

    return T


def weighted_cross_entropy(y_true,y_pred):
    '''
    Uses pre-computed global weighting factor, then weights cross-entropy 
    '''
    
    # get pre-computed class weights
    weights = K.backend.variable(CLASSWEIGHTS)

    #calculate cross entropy
    y_pred /= tf.reduce_sum(y_pred,axis=len(y_pred.get_shape())-1,keepdims=True)
    _epsilon = tf.convert_to_tensor(K.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
    #clip bad values
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    
    # calculate weighted loss per class and batch
    weighted_losses = y_true * tf.log(y_pred) * weights
       
    return -tf.reduce_sum(weighted_losses,len(y_pred.get_shape()) - 1)

def kl_divergence(y_true, y_pred):
    '''
    Calculates Kullback-Leibler divergence
    TODO: Fix this 
    '''
    epsilon = tf.convert_to_tensor(K.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
    #clip pred bad values
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    M = tf.ones_like(y_pred)
    M = tf.reduce_sum(M, axis=(1,2), keepdims=True)
    #caclulate frequency of each class
    y_true_prob = tf.divide(tf.reduce_sum(y_true,axis=(1,2),keepdims=True),M) #leave classes
    y_true_prob = tf.clip_by_value(y_true_prob, epsilon, 1. - epsilon)
    # y_true_prob = tf.Print(y_true_prob,[y_true_prob],message="p_c: ",summarize=10)

    #calculate divergence 
    cross_entropy = tf.reduce_sum(y_true_prob * tf.log(y_true_prob / y_pred),axis=(1,2,3))
    cross_entropy = tf.Print(cross_entropy,[tf.shape(cross_entropy)],message="NCE: ",summarize=10)

    return cross_entropy

def normalized_cross_entropy(y_true,y_pred):
    '''
    Modify original cross entropy by penalizing false-positive class
    '''
    epsilon = tf.convert_to_tensor(K.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
    #clip pred bad values
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)    
    #M = tf.constant(BATCHSIZE*H*W,dtype=y_pred.dtype.base_dtype)
    M = tf.ones_like(y_pred)
    M = tf.reduce_sum(M, axis=(1,2))

    #caclulate frequency of each class
    true_freq = tf.reduce_sum(y_true,axis=(1,2)) #leave classes
    false_freq = M - true_freq

    #calculate weight for each
    mask = tf.equal(tf.cast(true_freq,tf.int64),0) 
    true_weight = tf.where(mask, y=1/true_freq, x=tf.zeros_like(true_freq))
    
    mask = tf.equal(tf.cast(false_freq,tf.int64),0)
    false_weight = tf.where(mask, y=1/false_freq, x=tf.zeros_like(false_freq))

    #clip true bad values
    #y_true_prob = tf.clip_by_value(y_true_prob, epsilon, 1. - epsilon)
    #true_weight = tf.Print(true_weight,[true_weight],message="p_c: ",summarize=10)
    #false_weight = tf.Print(false_weight,[false_weight],message="p_i: ",summarize=10)

    
    #calculate predicted entropy
    pos_cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred) , axis=(1,2)) * true_weight
    neg_cross_entropy = -tf.reduce_sum((1 - y_true) * tf.log(1 - y_pred), axis=(1,2)) * false_weight 
    # pos_cross_entropy = tf.Print(pos_cross_entropy,[pos_cross_entropy[4]],message="CE_pos: ",summarize=16)
    # neg_cross_entropy = tf.Print(neg_cross_entropy,[neg_cross_entropy[4]],message="CE_neg: ",summarize=16)
    
    # calculate weighted loss per class and batch
    cross_entropy = tf.reduce_sum(pos_cross_entropy + neg_cross_entropy, axis = -1)
    #cross_entropy = tf.Print(cross_entropy,[tf.shape(cross_entropy)],message="NCE: ",summarize=10)
    
    return cross_entropy 


def cross_entropy_multiclass(y_true,y_pred):
    '''
    Modify original cross entropy by penalizing false-positive class
    '''
    epsilon = tf.convert_to_tensor(K.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
    
    # calculate maximum entropy of batch
    #Let M  be  the  total  number  of  words  in  the  STT  output,  
    # and  let m  be  the  number  of  those M words that are correct. 
    # Then the average probability that a word in the STT output will actually be  correct  is pc  =  m/M.  
    # As  suggested  by  the  earlier  discussions,  given pc  there  is  a  maximum value  for  
    # the  entropy  for  STT  actually  getting  the  words  correct  vs.  incorrect,  
    # and  we  multiply that entropy by M giving a value that we will call Hmax :
    # Hmax =  -m*log2(pc)-(M-m)*log2(1-pc) 
    
    #caclulate frequency of each class
    freq = tf.reduce_sum(y_true,axis=(0,1,2)) #leave classes
    #calculate average probability of each class
    M = tf.constant(BATCHSIZE*H*W,dtype=y_pred.dtype.base_dtype)
    y_true_prob = freq / (M)
    #clip true bad values
    y_true_prob = tf.clip_by_value(y_true_prob, epsilon, 1. - epsilon)
    #y_true_prob = tf.Print(y_true_prob,[y_true_prob],message="p_c: ",summarize=10)

    #calculate Hmax
    max_cross_entropy = -freq * tf.log(y_true_prob) - (M - freq) * tf.log(1 - y_true_prob)
    max_cross_entropy = tf.Print(max_cross_entropy,[max_cross_entropy],message="h_max: ",summarize=10)

    #clip pred bad values
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    #calculate predicted entropy
    pred_cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred),axis=(0,1,2)) - tf.reduce_sum((1 - y_true) * tf.log(1 - y_pred),axis=(0,1,2))
    pred_cross_entropy = tf.Print(pred_cross_entropy,[pred_cross_entropy],message="h_pred: ",summarize=10)

    # calculate weighted loss per class and batch
    cross_entropy = (pred_cross_entropy - max_cross_entropy)/max_cross_entropy
    cross_entropy = tf.Print(cross_entropy,[cross_entropy],message="NCE: ",summarize=10)
    # cross_entropy = tf.Print(cross_entropy,[tf.shape(cross_entropy)],message="CE: ",summarize=8)
    #loss = -tf.reduce_sum(cross_entropy,len(y_pred.get_shape()) - 1)
    # loss = tf.Print(loss,[tf.shape(loss)],message="Loss: ",summarize=8)
    cross_entropy = tf.reduce_mean(cross_entropy)

    return cross_entropy

def weighted_batch_cross_entropy(y_true,y_pred):
    '''
    Calculates frequency of per batch, then weights cross-entropy accordingly
    TODO: finish this function
    '''
    
    # calculate frequency for each class
    freq = tf.reduce_sum(y_true,axis=(0,1,2)); #leave classes

    #calculate max freq
    max_freq = tf.reduce_max(freq)

    #normalize it by max
    freq_norm = max_freq/(freq+1); #avoid zero division

    # find classes that don't contribute
    w_mask = tf.equal(tf.cast(freq_norm,tf.int64),tf.cast(max_freq,tf.int64))
    weights = tf.where(w_mask,y=freq_norm,x=tf.ones_like(freq_norm))

    #take a square of weights to smooth
    weights = tf.pow(weights,0.3) #was working fine 0.2
    #print_weights = tf.Print(weights,[weights]) #print statement

    #calculate cross entropy
    y_pred /= tf.reduce_sum(y_pred,axis=len(y_pred.get_shape())-1,keep_dims=True)
    _epsilon = tf.convert_to_tensor(K.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
    #clip bad values
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    
    # calculate weighted loss per class and batch
    weighted_losses = (y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))*weights
       
    return -tf.reduce_sum(weighted_losses,len(y_pred.get_shape()) - 1)

def load_json_model(modelName):
    filePath = './checkpoint/'+modelName+".json";
    fileWeight = './checkpoint/'+modelName+"_weights.h5"

    with open(filePath,'r') as fp:
        json_data = fp.read();
    model = model_from_json(json_data,
                        custom_objects={'ElementWiseSum':ElementWiseSum,
                                'BilinearUpsampling':BilinearUpsampling,
                                'BilinearInterpolation':BilinearInterpolation}
                )
    model.load_weights(fileWeight)

    return model


def metric_per_label(label,alpha,beta):

    if alpha==beta==1:
        def jaccard(y_true,y_pred):
            y_true = K.backend.argmax(y_true,axis=-1);
            y_pred = K.backend.argmax(y_pred,axis=-1);

            true = K.backend.cast(K.backend.equal(y_true,label),'float32');
            pred = K.backend.cast(K.backend.equal(y_pred,label),'float32');

            return tversky_score(true,pred,alpha,beta)

        return jaccard
    elif alpha==beta==0.5:
        def dice(y_true,y_pred):
            y_true = K.backend.argmax(y_true,axis=-1);
            y_pred = K.backend.argmax(y_pred,axis=-1);

            true = K.backend.cast(K.backend.equal(y_true,label),'float32');
            pred = K.backend.cast(K.backend.equal(y_pred,label),'float32');

            return tversky_score(true,pred,alpha,beta)
        return dice
    else:
        def f1(y_true,y_pred):
            y_true = K.backend.argmax(y_true,axis=-1);
            y_pred = K.backend.argmax(y_pred,axis=-1);

            true = K.backend.cast(K.backend.equal(y_true,label),'float32');
            pred = K.backend.cast(K.backend.equal(y_pred,label),'float32');
            return tversky_score(true,pred,alpha,beta)
        
        return f1

from keras.callbacks import LearningRateScheduler
def step_decay_schedule(initial_lr=1e-5, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)



def train_model(trainGen,valGen,stepsPerEpoch,numEpochs,valSteps):

    try:
        model = load_json_model(modelName)
        print("Loading model...");
    except Exception as e:
        print(e);
        print("Creating new model...")
        # model = UNet1024()
        # model = UNet1024SkipConnection()
        model = MultiScaleNet()
        # model = DilatedFCN()


    losses = {
    # "organ_output": normalized_cross_entropy
    #"organ_output":dist_loss
    "organ_output": soft_dice_loss
    # "organ_output": kl_divergence
    # "organ_output": "categorical_crossentropy"
    # "organ_output": weighted_cross_entropy
    #"organ_output": cross_entropy_multiclass
    # "organ_output": weighted_batch_cross_entropy
    }
    lossWeights = {
    "organ_output": 1.0
    }

    print(model.summary())
    lr_finder = LRFinder(min_lr=1e-5, 
                        max_lr=1e-2, 
                        steps_per_epoch=np.ceil(NUMEPOCHS/BATCHSIZE), 
                        epochs=3)

    lr_anneal = LRAnnealer(initial_lr=LEARNRATE, decay_rate=DECAYRATE, exponetial=False)

    optimizer = K.optimizers.Adam(
            lr = LEARNRATE, #decay = DECAYRATE        
            )
    
    esophagus_dice = metric_per_label(1,alpha=0.5,beta=0.5)
    heart_dice = metric_per_label(2,alpha=0.5,beta=0.5)
    trachea_dice = metric_per_label(3,alpha=0.5,beta=0.5)
    aorta_dice = metric_per_label(4,alpha=0.5,beta=0.5)
    


    #compile model
    model.compile(optimizer=optimizer,
                    loss = losses,#tot_loss,
                    loss_weights=lossWeights,
                    metrics=[esophagus_dice,heart_dice,trachea_dice,aorta_dice]
                    );

    #define callbacks
    modelCheckpoint = K.callbacks.ModelCheckpoint("./checkpoint/"+modelName+"_weights.h5",
                                'val_loss',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='min', period=1)

    reduceLearningRate = K.callbacks.ReduceLROnPlateau(monitor='loss',
                                factor=0.5, patience=5,
                                verbose=1, mode='auto',
                                cooldown=1, min_lr=0)

    earlyStopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                patience=3,
                                verbose=1,
                                min_delta = 0.0001,mode='min')
    validationMetric = Metrics(valGen,valSteps,BATCHSIZE)

        
    #save only model
    with open('./checkpoint/'+modelName+'.json','w') as fp:
        fp.write(model.to_json());

    #fit model and store history
    hist = model.fit_generator(trainGen, 
              steps_per_epoch = stepsPerEpoch,
              epochs = numEpochs,
              class_weight = 'auto',
              validation_data = valGen,
              validation_steps = valSteps,
              verbose=1,
            #   callbacks=[lr_anneal, validationMetric,modelCheckpoint]
              callbacks = [ validationMetric,modelCheckpoint]
              )
    
    #lr_finder.plot_loss()

    return hist
