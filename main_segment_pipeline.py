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
        imgBatch = dataGenerator.denormalize(imgBatch)

        #gen output is dictionary
        labelBatch = label["organ_output"]
                
        print(imgBatch.shape, labelBatch.shape)
        #plot label and data
        k = plotter.plot_slice_label(imgBatch,labelBatch)
        if k==27: break


def morphological_closing(img):
    kernel = np.ones((5,5),np.uint8);

    if len(img.shape)==2:
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        outImg = np.zeros_like(img);
        for idx in range(img.shape[-1]):
            #outImg[...,idx] = cv2.GaussianBlur(img[...,idx],(7,7),5);
            outImg[...,idx] = cv2.morphologyEx(img[...,idx], cv2.MORPH_CLOSE, kernel)
            #outImg[...,idx] = cv2.morphologyEx(img[...,idx], cv2.MORPH_OPEN, kernel)
            #outImg[...,idx] = cv2.morphologyEx(img[...,idx], cv2.MORPH_DILATION, kernel)
        return outImg

def plot_prediction(valGen,model,normalize):

    ax = None; idx = 0; imgNum = 0;
    
    while imgNum<VALSIZE:
        data, label = next(valGen)

        #apply inverse normalization to image
        if normalize is not None:
            std = normalize["vars"];
            means = normalize["means"];
            #get non zero
            nonZero = np.ma.masked_equal(data[idx].squeeze(),0);
            image = (nonZero*std+means).data;
        else:
            image = data[idx,...,0];
        
        if label["organ_output"].shape[-1]>1:
            true_organ = label["organ_output"][idx,...].argmax(axis=-1);
        else:
            true_organ = label["organ_output"][idx,...,0];            

        #apply morphological tranformation
        true_organ = morphological_closing(true_organ.astype('uint8'))

        #normalize true labels and input
        image = (image-image.min())/(image.max()-image.min());
        #true_organ = true_organ.astype("float32")/true_organ.max();

        #predict image and normalize first
        pred_organ = model.predict(data);

        #check if prediction is multi-label
        if pred_organ.shape[-1]>1:
            pred_organ = pred_organ.argmax(axis=-1);
        else:
            pred_organ = pred_organ[...,0];
        

        #get selected index of predictions
        pred_organ = pred_organ[idx,...]
        pred_organ = morphological_closing(pred_organ.astype('uint8'))
        print(np.unique(pred_organ))
        
        #normalize predicted labels
        #pred_organ = pred_organ.astype("float32")/(pred_organ.max());

        #imgStack = np.hstack([true_organ, pred_organ,image]);
        #cv2.imshow("Predicted Expected Results",imgStack);
        #k = cv2.waitKey(0);
        #if k==27:
        #    break        
        
        ax = overlay_contours_plot(pred_organ,true_organ,image,imgNum,ax); 
        #overlay_contours_save(pred_organ,true_organ,image,imgNum);         
        imgNum+=1;

import matplotlib.pyplot as plt
def overlay_contours_plot(pred,truth,image,imgNum,ax=None):
    colors = ('g','r','c','m','b','y');

    if ax is None:
        fig, axes = plt.subplots(ncols=2, figsize=(15,8), gridspec_kw = {'wspace':0, 'hspace':0,'left':0,'right':1.0,'top':0.95,'bottom':0.});
        ax = axes.ravel();

    ax[0].set_title("True contours");
    ax[0].imshow(image, cmap="gray");
    
    for idx in [1,2,3,4,5,6]: 
        ax[0].contour((truth==idx).astype('float32'), colors=colors[idx-1], normlinestyles='solid', linewidths=1);

    ax[0].axis('off');

    ax[1].set_title("Predicted contours");    
    ax[1].imshow(image, cmap="gray");
    for idx in [1,2,3,4,5,6]:
        ax[1].contour((pred==idx).astype('float32'), colors=colors[idx-1], linestyles='solid', linewidths=1);
    ax[1].axis('off');    

    #plt.tight_layout();
    plt.show(0);
    plt.pause(1);
    
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
        y_true = y_true.argmax(axis=-1);
        y_pred = y_pred.argmax(axis=-1);

        #find matching labels
        true = (y_true==label).astype('float32');
        pred = (y_pred==label).astype('float32');

        return tversky_score_numpy(true,pred,alpha,beta)
    return wrapper;

def esophagus_dice():
    return metric_decorator(1,0.5,0.5)
def heart_dice():
    return metric_decorator(2,0.5,0.5)
def trachea_dice():
    return metric_decorator(3,0.5,0.5)
def aorta_dice():
    return metric_decorator(4,0.5,0.5)

import time    
def report_validation_results(testFile,model,batchSize=1024,steps=int(1024/128)+1):

    valGen = data_generator(testFile, batchSize=batchSize, augment=False, normalize={"means":imgMean,"vars":imgStd},shuffle=False)
    step = 0; 
    scores = [];

    while step<1:
        X_batch, Y_batch = next(valGen);
        Y_true = Y_batch['organ_output'];
        step+=1;
        start = time.time();
        Y_pred = model.predict(X_batch);
        end = time.time();
        print(X_batch.shape, Y_true.shape, step);

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
    trainFile = "TRAIN.h5";
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
    #trainGen = data_generator(trainFile,batchSize=BATCHSIZE,augment=True,normalize={"means":imgMean,"vars":imgStd},shuffle=True)
    trainGen = dataGenerator.generate_data(trainFile,
                            batchSize=BATCHSIZE,augment=False,shuffle=True)
    #create test data generator
    valGen = dataGenerator.generate_data(testFile,
                            batchSize=BATCHSIZE,augment=False,shuffle=False)
     

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
        ax[1].plot(history["dice_1"],"r*",history["dice_2"],"g^",history["dice_3"],"b>",history["dice_4"],"k<",history["dice_5"],"mo");
        ax[1].legend(labels=["Bladder","Rectum","Femoral Head","CTV","Sigmoid"]);
        ax[1].set_title("Dice coefficients vs epochs")
        ax[1].set_xlabel("# of epochs");
        fig.savefig("./log/"+modelName+".jpg")

        
    elif arg=='test':
        model = load_json_model(modelName)
        plot_prediction(valGen,model,normalize={"means":imgMean,"vars":imgStd})

    elif arg=='report':
        model = load_json_model(modelName)
        report_validation_results(testFile,model)

    elif arg=='submit':
        #load model
        model = load_json_model(modelName)
        #creat instance of SubmitPrediction
        predictor = SubmitPrediction(
                            pathToImages='../../SegmentationDataSets/SegTHOR/',
                            #filePattern='GT',
                            filePattern='Patient'
                            )
        predictor.set_model(model)
        predictor.set_normalization(normParam={"means":imgMean,"vars":imgStd})
        predictor.predict_nii_patients(batchSize=8)      

    elif arg=='plot':
        test_generator(valGen,dataGenerator)
        #test_generator(trainGen,dataGenerator)
