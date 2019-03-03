#import everything
import os, sys, glob
import numpy as np

from helper_data_generator import *
from helper_to_train import *
from helper_to_plot import Plotter
from helper_to_submit import SubmitPrediction


def test_generator(myGen,dataGenerator):

    #create instance of Plotter class
    plotter = Plotter()

    while True:
        #images are already processed in 
        imgBatch, label = next(myGen)

        #denormalize image for plotting
        imgDeNorm = dataGenerator.denormalize(imgBatch)
        imgDeZoom = dataGenerator.unzoom(imgDeNorm)
        imgDeCrop = dataGenerator.uncrop(imgDeZoom)

        #gen output is dictionary
        labelBatch = label["organ_output"]
                
        print(imgBatch.shape, labelBatch.shape)
        #plot label and data
        #k = plotter.plot_slice_label(imgBatch,labelBatch)
        k = plotter.plot_slices([dataGenerator.images[0], imgDeCrop[0]])
        if k==27: break


def plot_prediction(valGen,model,dataGenerator):

    ax = None; idx = 0; imgNum = 0;
    
    #create instance of Plotter class
    plotter = Plotter()

    while imgNum<VALSIZE:
        imgBatch, label = next(valGen)
        labelBatch = label["organ_output"]

        #denormalize image for plotting
        imgBatchDeNorm = dataGenerator.denormalize(imgBatch)       

        #predict image and normalize first
        labelPred = model.predict(imgBatch)
        print(labelPred.shape)

        #get selected index of predictions
        labelPredMorph = dataGenerator.morphological_operation_3d(labelPred)
        # labelPredMorph = dataGenerator.morphological_operation(labelPred,'open')

        # k = plotter.plot_label_prediction(imgBatchDeNorm[0],labelBatch,labelPredMorph[0].argmax(axis=-1))
        # if k==27: break
        
        for idx in range(imgBatch.shape[0]):
            if np.max(labelBatch.argmax(-1))>0:
                ax = overlay_contours_plot(labelPredMorph[idx].argmax(axis=-1),labelBatch[idx].argmax(-1),imgBatchDeNorm[idx],imgNum,ax); 
        # if np.max(labelBatch.argmax(-1))>0:
        #     overlay_contours_save(labelPred[0],labelBatch[0].argmax(-1),imgBatchDeNorm[0],imgNum);         
        imgNum+=imgBatch.shape[0]

import matplotlib.pyplot as plt
def overlay_contours_plot(pred,truth,image,imgNum,ax=None):
    colors = ('g','r','c','m','b','y');

    if ax is None:
        fig, axes = plt.subplots(ncols=2, figsize=(15,8), gridspec_kw = {'wspace':0, 'hspace':0,'left':0,'right':1.0,'top':0.95,'bottom':0.});
        ax = axes.ravel();

    ax[0].set_title("True contours")
    ax[0].imshow(image, cmap="gray")
    
    for idx in [1,2,3,4,5,6]: 
        ax[0].contour((truth==idx).astype('float32'), colors=colors[idx-1], normlinestyles='solid', linewidths=1)

    ax[0].axis('off')

    ax[1].set_title("Predicted contours")    
    ax[1].imshow(image, cmap="gray")
    for idx in [1,2,3,4,5,6]:
        ax[1].contour((pred==idx).astype('float32'), colors=colors[idx-1], linestyles='solid', linewidths=1)
    ax[1].axis('off')    

    #plt.tight_layout();
    plt.show(0)
    plt.pause(0.5)
    
    ax[1].clear(); ax[0].clear();        

    return ax

def overlay_contours_save(pred,truth,image,imgNum):
    

    fig = plt.figure(figsize=(8, 6));
    colors = ('g','r','c','m','b','y');

    
    sub1 = fig.add_subplot(121)
    sub1.imshow(image, cmap="gray");
    
    for idx in [1,2,3,4,5,6]: 
        sub1.contour((truth==idx).astype('float32'), colors=colors[idx-1], normlinestyles='solid', linewidths=0.7);

    sub1.axis('off');

    sub2 = fig.add_subplot(122)
    sub2.imshow(image, cmap="gray");
    for idx in [1,2,3,4,5,6]:
        sub2.contour((pred==idx).astype('float32'), colors=colors[idx-1], linestyles='solid', linewidths=0.7);
    sub2.axis('off');

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.0, left=0.0, top = 1, right=1, wspace=0, hspace=0)
    plt.pause(1);

    plt.savefig('./figures/val_seg_'+str(imgNum)+".jpg", dpi=400, bbox_inches='tight');
    plt.close("all");

def get_normalization_param(trainFile):
    with h5py.File(trainFile,"r") as fp:
        means = fp["means"][0];
        variance = fp["variance"][0]
    return means, variance 

        
def tversky_score_numpy(y_true, y_pred, alpha = 0.5, beta = 0.5):

    true_positives = y_true * y_pred;
    false_negatives = y_true * (1 - y_pred);
    false_positives = (1 - y_true) * y_pred;

    num = true_positives.sum(axis = (1,2)) #compute loss per-batch
    den = num+alpha*false_negatives.sum(axis = (1,2)) + beta*false_positives.sum(axis=(1,2))+1
    #remove zero measurements from sample
    mask = (num==0)
    T = (num/den);
    return sum(~mask), T[~mask].mean()


def metric_decorator(label,alpha,beta):

    def wrapper(y_true,y_pred):
        #convert one-hot to labels 
        y_true = y_true.argmax(axis=-1)
        y_pred = y_pred.argmax(axis=-1)

        #find matching labels
        true = (y_true==label).astype('float32')
        pred = (y_pred==label).astype('float32')

        return tversky_score_numpy(true,pred,alpha,beta)
    return wrapper

def esophagus_dice():
    return metric_decorator(1,0.5,0.5)
def heart_dice():
    return metric_decorator(2,0.5,0.5)
def trachea_dice():
    return metric_decorator(3,0.5,0.5)
def aorta_dice():
    return metric_decorator(4,0.5,0.5)

import time    
def report_validation_results(testFile,model,dataGenerator,batchSize=1024,steps=int(1024/128)+1):

    #create data generator
    valGen = dataGenerator.generate_data(testFile, batchSize=batchSize, augment=False,shuffle=False)
    step = 0; 
    scores = [];

    while step<1:
        X_batch, Y_batch = next(valGen)
        Y_true = Y_batch['organ_output']
        step+=1
        start = time.time()
        Y_pred = model.predict(X_batch)
        
        #Test effect of morphological operations
        # labelPredMorph = dataGenerator.morphological_operation_3d(labelPred)
        Y_pred = dataGenerator.morphological_operation(Y_pred,'open')
        #
        end = time.time()
        print(X_batch.shape, Y_true.shape, step)

        #calculated weighted scores
        scores.append(('esophagus',esophagus_dice()(Y_true,Y_pred)[1]));  
        scores.append(('heart',heart_dice()(Y_true,Y_pred)[1]));
        scores.append(('trachea',trachea_dice()(Y_true,Y_pred)[1]));
        scores.append(('aorta',aorta_dice()(Y_true,Y_pred)[1]));
        
    print("Time took to predict {0} slices is {1} seconds".format(batchSize,end-start));
    print(scores)

import pickle
if __name__=='__main__':
    
    arg = sys.argv[1]
    gpu = sys.argv[2]

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

   
    #define train file name
    trainFile = "TRAIN_MERGED.h5";
    testFile = "TEST.h5";

    #now get normalization parameters
    imgMean, imgStd = get_normalization_param(trainFile.replace(".h5","_STAT.h5"))

    #create data generator
    dataGenerator = DataGenerator(normalize={"means":imgMean,"vars":imgStd})

    #create train data generator using stratified
    #trainGen = data_generator_stratified(trainFile,batchSize=BATCHSIZE,augment=True,
    #        normalize={"means":imgMean,"vars":imgStd}
    #        )
    #create train data generator
    trainGen = dataGenerator.generate_data(trainFile,
                            batchSize=BATCHSIZE,
                            augment=True,shuffle=True)
    #create test data generator
    valGen = dataGenerator.generate_data(testFile,
                           batchSize=8,
                        #    batchSize=BATCHSIZE,
                           augment=False,shuffle=False)
     

    arg = sys.argv[1]
    if arg=='train':
        hist = train_model(trainGen, valGen, STEPPEREPOCHS, NUMEPOCHS, VALSTEPS);

        with open('./log/'+modelName+".log", 'wb') as fp:
            pickle.dump(hist.history, fp)
        history = hist.history;

        fig, axes = plt.subplots(ncols=2,nrows=1)
        ax = axes.ravel();
        ax[0].plot(history['loss'],'r*',history['val_loss'],'g^');
        ax[0].legend(labels=["Train loss","Val Loss"],loc="best");
        ax[0].set_title("Cross entropy loss vs epochs")        
        ax[0].set_xlabel("# of epochs");        
        ax[1].plot(history["dice_1"],"r*",history["dice_2"],"g^",history["dice_3"],"b>",history["dice_4"],"k<");
        ax[1].legend(labels=["Bladder","Rectum","Femoral Head","CTV","Sigmoid"]);
        ax[1].set_title("Dice coefficients vs epochs")
        ax[1].set_xlabel("# of epochs");
        fig.savefig("./log/"+modelName+".jpg")

        
    elif arg=='test':
        model = load_json_model(modelName)
        plot_prediction(valGen,model,dataGenerator)

    elif arg=='report':
        model = load_json_model(modelName)
        report_validation_results(testFile,model,dataGenerator)

    elif arg=='submit':
        #load model
        model = load_json_model(modelName)
        #creat instance of SubmitPrediction
        predictor = SubmitPrediction(
                            #pathToImages='../../SegmentationDataSets/SegTHOR/', #all files 
                            pathToImages='../../SegmentationDataSets/SegTHOR/validation',
                            filePattern='Patient'
                            )
        predictor.set_model(model)
        predictor.set_normalization(normParam={"means":imgMean,"vars":imgStd})
        predictor.predict_nii_patients(batchSize=4)      

    elif arg=='plot':
        test_generator(valGen,dataGenerator)
        # test_generator(trainGen,dataGenerator)
