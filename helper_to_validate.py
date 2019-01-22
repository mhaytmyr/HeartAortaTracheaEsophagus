import os,sys
import numpy as np
import cv2
import nibabel as nib
from helper_to_processing import *

import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def plot(self,imgInput=None,labelInput=None,predInput=None):
        if imgInput is None:
            #nothing to plot; raise error
            sys.exit(type(self).__name__+".plot() imgInput has to be provided ")
        elif labelInput is None:
            #only plot imgInput
            return self.plot_slice(imgInput)
        elif predInput is None:
            #show img and ground truth
            return self.plot_slice_label(imgInput,labelInput)
        else:
            #show both pred and ground truth
            return self.plot_label_prediction(imgInput,imgInput,labelInput)

    def plot_slices(self,imgInput):
        #imgInput is list of images
        imgStack = []
        for idx in range(len(imgInput)):
            print(type(self).__name__+".plot_slices img shape"+str(imgInput[idx].shape))
            imgStack.append(self.tensor_to_image_wrapper(imgInput[idx]))

        imgNorm = np.hstack(imgStack)
        cv2.imshow("Slice ",imgNorm)
        k = cv2.waitKey(0);
        return k

    def tensor_to_image_wrapper(self,imgInput=None):
        #Method to convert 4D array image to 2D image
        #first remove extra dimentions
        imgInput = imgInput.squeeze()
        #if 2D image nothing to do; return image
        if len(imgInput.shape)==2:
            imgNorm = (imgInput-imgInput.min())/imgInput.max()
        elif len(imgInput.shape)==3:
            idx = np.random.randint(0,imgInput.shape[0]-1)
            imgNorm = (imgInput[idx]-imgInput[idx].min())/imgInput[idx].max()
        else:
            sys.exit(type(self).__name__+".tensor_to_image_wrapper() can't convert input to image "+str(imgInput.shape)) 
        return imgNorm

    def tensor_to_label_wrapper(self,labelInput=None):
        #TODO: method to convert 4D label array to 2D image'
        pass

    def plot_slice(self,imgInput):
        #method to plot image slice only
        #first denormalize image; 
        imgNorm = self.tensor_to_image_wrapper(imgInput)
        cv2.imshow("Slice ",imgNorm)
        k = cv2.waitKey(0)
        return k


    def plot_slice_label(self,imgInput,labelInput):
        #TODO: method to plot image and label masks
        pass

    def plot_label_prediction(self,imgInput,labelInput,predInput):
        #TODO: method to compare ground truth and prediction contourd
        pass


    
class SubmitPrediction:
    def __init__(self,pathToImages,filePattern='Patient'):
        self.images = self.get_list_of_images(pathToImages,filePattern)
        self.model = None
        self.normParam = None

    def set_normalization(self,normParam):
        self.normParam = normParam

    def set_model(self,model):
        self.model = model

    #method to store list of images 
    def get_list_of_images(self,pathToImages,filePattern):
        #assume images are nii
        #folders = [os.path.join(pathToImages, name) for name in os.listdir(pathToImages) if os.path.isdir(os.path.join(pathToImages,name))];
        
        #find endpoints
        endpoints = [subdir for subdir,dirs,files in os.walk(pathToImages) if len(dirs)== 0];
        
        #find files matching description
        files = []
        for path in endpoints:
            for item in os.listdir(path):
                if filePattern in item:
                    files.append('/'.join([path,item]))                   
        
        #return directory names
        return files

    #method to run all patients
    def predict_nii_patients(self,batchSize=16):
        
        #initialize ImageProcessor class
        processor = ImageProcessor(self.normParam)    

        #initialize ImagePlotter class
        plotter = Plotter()

        #get batch of images
        for item in self.images:
            
            print("Predicting...",item)

            #load patient image
            slices = nib.load(item);   
                   
            #predict image patch
            k = self.predict_image_slices(slices,batchSize,processor,plotter)
            if (k==27):
                break
        pass            


    #method to extract slices from image batches and predict
    def predict_image_slices(self,slices,batchSize=16,processor=None,plotter=None):
        
        #initialize pre-processor class
        if processor is None:
            sys.exit(type(self).__name__+".predict_image_slices() needs ImageProcessor class")

        #apply preprocessing
        imgStandard = processor.standardize_img(slices)
        imgCrop = processor.crop(imgStandard)
        imgZoom = processor.zoom(imgCrop)
        imgNorm = processor.normalize(imgZoom)
        
        #apply inverse preprocessing
        imgDeNorm = processor.denormalize(imgNorm)
        imgUnZoom = processor.unzoom(imgDeNorm)
        imgDeCrop = processor.uncrop(imgUnZoom)

        
        n = imgNorm.shape[0]

        for idx in range(0,n,batchSize):
            #get batch of image
            #imgBatch = imgInput[idx:idx+batchSize] 
            imgBatch = imgDeNorm[idx:idx+batchSize]
            
            imgZoomUnZoom = [imgCrop[idx],imgUnZoom[idx]]
            #print(abs(imgCrop[idx]-imgUnZoom[idx]).max())
            imgCropUnCrop = [imgStandard[idx],imgDeCrop[idx],imgStandard[idx]-imgDeCrop[idx]]

            print(idx,idx+batchSize,imgBatch.shape)
            #k = plotter.plot(imgBatch)
            #k = plotter.plot_slices(imgCropUnCrop)
            k = plotter.plot_slices(imgZoomUnZoom)
            if (k==27) or (k=='n'):
                return k
        return k
                
