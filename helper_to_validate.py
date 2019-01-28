import os,sys,time
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

    def tensor_to_image_wrapper(self,imgInput=None,idx=None):
        #Method to convert 4D array image to 2D image
        #first remove extra dimentions
        imgInput = imgInput.squeeze()
        #if 2D image nothing to do; return image
        if len(imgInput.shape)==2:
            imgNorm = (imgInput-imgInput.min())/imgInput.max()
        elif len(imgInput.shape)==3:
            if idx is None:
                idx = np.random.randint(0,imgInput.shape[0]-1)
            imgNorm = (imgInput[idx]-imgInput[idx].min())/imgInput[idx].max()
        else:
            sys.exit(type(self).__name__+".tensor_to_image_wrapper() can't convert input to image "+str(imgInput.shape)) 
        return imgNorm

    def tensor_to_label_wrapper(self,labelInput=None,idx=None):
        #Method to convert 4D label array to 2D image'
        labelInput = labelInput.squeeze()
        #apply argmax to last index, since prediction is one-hot encoded
        if len(labelInput.shape)==4:
            labelInput = labelInput.argmax(axis=-1) 
        print(np.unique(labelInput))
        return self.tensor_to_image_wrapper(labelInput,idx)

    def plot_slice(self,imgInput):
        #method to plot image slice only
        #first denormalize image; 
        imgNorm = self.tensor_to_image_wrapper(imgInput)
        cv2.imshow("Slice ",imgNorm)
        k = cv2.waitKey(0)
        return k


    def plot_slice_label(self,imgInput,labelInput):
        #Method to plot image and label masks
        #input is 4D array, so we need to choose one
        idx =  np.random.randint(0,imgInput.shape[0]-1)
        img = self.tensor_to_image_wrapper(imgInput,idx=idx)
        label = self.tensor_to_label_wrapper(labelInput,idx=idx)

        imgNorm = np.hstack([img,label])
        cv2.imshow("Slice ",imgNorm)
        k = cv2.waitKey(0)
        return k

    def plot_label_prediction(self,imgInput,labelInput,predInput):
        #Method to compare ground truth and prediction contours
        pass


    
class SubmitPrediction:
    def __init__(self,pathToImages,filePattern='Patient'):
        self.imageList = self.get_list_of_images(pathToImages,filePattern)
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
        for item in self.imageList:
            print("Predicting...",item)
            #load patient image
            slices = nib.load(item);   
            #predict image patch
            #k = self.save_truth_label(slices,processor,plotter,item)
            k = self.predict_image_slices(slices,batchSize,processor,plotter,item)
            #k = self.processor_tester(slices,batchSize,processor,plotter)
            if (k==27):
                break

    def save_prediction_as_nii(self,predInput,inputFile):
        '''
        Method to save prediction as nii file
        input: 3D array in the format (N,H,W)
        output: None
        '''
        #1. for submission I need to transpose axis
        labelTrans = np.transpose(predInput,[1,2,0]).astype('uint8')
        #2. create destination file name and affine
        affine = nib.load(inputFile).get_affine()
        hdr = nib.load(inputFile).get_header()

        patName = inputFile.split('/')[-1].replace('.gz','')
        dstFile = './result/'+patName
        print(labelTrans.shape, dstFile)
        #3. save image as nifti
        niiObj = nib.Nifti1Image(labelTrans,affine=affine,header=hdr)
        print(niiObj.shape)
        nib.save(niiObj,dstFile)

    #method to extract slices from image batches and predict
    def save_truth_label(self,slices,processor=None,plotter=None,patient=None):
        #this is ground truth so create different name
        patName = patient.split('/')[-2]+'.nii.gz'
        patient = patient.replace('GT.nii.gz',patName)

        labelStandard = processor.standardize_label(slices)
        print(labelStandard.shape)
        labelDeStandard = processor.de_standardize_nii(labelStandard)
        print(labelDeStandard.shape)
        #saving prediction
        self.save_prediction_as_nii(labelDeStandard,patient)
        return 0

    #method to extract slices from image batches and predict
    def predict_image_slices(self,slices,batchSize=16,processor=None,plotter=None,patient=None):
        
        #initialize pre-processor class
        if processor is None:
            sys.exit(type(self).__name__+".predict_image_slices() needs ImageProcessor class")

        t0 = time.time()
        #1. pre-process image by cropping, zooming and normalization
        imgNorm = processor.pre_process(slices)
        print("Pre-processing of {0} took {1:.4f} s".format(slices.shape,time.time()-t0))
        #2. predict on current batch
        t0 = time.time()
        labelPred = self.model.predict(processor.img_to_tensor(imgNorm))
        print("Inference of {0} took {1:.4f} s".format(imgNorm.shape,time.time()-t0))
        #3. prediction has (256,384) -> convert back to original crop size
        t0 = time.time()
        labelPredUnZoom = processor.unzoom_label(labelPred)
        print("Unzooming took {0:.4f} s".format(time.time()-t0))
        #4. pad uncropped label to have (512,512) size
        t0 = time.time()
        labelPredDeCrop = processor.uncrop_label(labelPredUnZoom)
        print("Padding took {0:.4f} s".format(time.time()-t0))
        #5. this particular dataset was rotated -90 degree, so need to fix it
        t0 = time.time()
        labelPredDeStandard = processor.de_standardize_nii(labelPredDeCrop.argmax(axis=-1))
        print("Unzooming took {0:.4f} s".format(time.time()-t0))

        #saving prediction
        self.save_prediction_as_nii(labelPredDeStandard,patient)
        return 0


    def processor_tester(self,slices,batchSize=6,processor=None,plotter=None):
        #apply preprocessing
        imgStandard = processor.standardize_img(slices)
        imgDeStandard = processor.de_standardize_nii(imgStandard)
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
            imgBatch = imgNorm[idx:idx+batchSize]
            
            #imgZoomUnZoom = [imgCrop[idx],imgUnZoom[idx]]
            #print(abs(imgCrop[idx]-imgUnZoom[idx]).max())
            #imgCropUnCrop = [imgStandard[idx],imgDeCrop[idx],imgStandard[idx]-imgDeCrop[idx]]
            #imgStandUnStand = [imgStandard[idx],imgDeStandard[...,idx]]
            
            #labelPred = self.model.predict(processor.img_to_tensor(imgBatch))
            #labelPredUnZoom = processor.unzoom_label(labelPred)
            #labelPredDeCrop = processor.uncrop_label(labelPredUnZoom)
            #labelPredDeStandard = processor.de_standardize_nii(labelPredDeCrop.argmax(axis=-1))
            print(idx,idx+batchSize,imgBatch.shape)
            
            #k = plotter.plot_slice_label(imgDeNorm[idx:idx+batchSize],labelPred)
            #k = plotter.plot_slices(imgCropUnCrop)
            k = plotter.plot_slices([imgDeStandard[idx],processor.images[idx]])
            if (k==27) or (k=='n'):
                return k
        return k
