import os,sys,time
import numpy as np

import nibabel as nib
from helper_to_processing import *
from helper_to_plot import Plotter

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
        #hdr = nib.load(inputFile).get_header()

        patName = inputFile.split('/')[-1].replace('.gz','')
        dstFile = './result/'+patName
        print(labelTrans.shape, dstFile)
        #3. save image as nifti
        niiObj = nib.Nifti1Image(labelTrans,affine=affine)
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
        imgNorm = processor.pre_process_img(slices)
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
        labelPredMorph = processor.morphological_operation(labelPredDeStandard.astype(np.uint8))
        print("Unzooming took {0:.4f} s".format(time.time()-t0))

        #saving prediction
        self.save_prediction_as_nii(labelPredMorph,patient)
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
