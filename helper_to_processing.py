import os, sys, glob, pdb
import numpy as np
import nibabel as nib
import pydicom

import cv2, pdb
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

from config import *


class Normalizer:
    def __init__(self,normalize=None):
        self.normalization = normalize
        self.images = None #original input images
        self.labels = None #original input labels

    def normalize(self,inputImg):
        '''
        Method to normalize image using predtermined normalization factor
        input: ndarray
        output: ndarray (normalized)
        '''
        nonZero = np.ma.masked_equal(inputImg,0)
        normalized = ((nonZero-self.normalization["means"])/self.normalization["vars"]).data
        return normalized

    def normalize_new(self,inputImg):
        '''
        Method to normalize image using range of CT numbers
        input: ndarray
        output: ndarray (normalized btw [0,1])
        '''
        normalized = inputImg/1000. - 2
        return normalized

    def denormalize_new(self,inputImg):
        '''
        Method to normalize image using range of CT numbers
        input: ndarray
        output: ndarray (normalized btw [0,4000])
        '''
        denorm = (inputImg + 2) * 1000.
        return denorm

    def denormalize(self,inputImg):
        '''
        Method to apply inverse normalization to image
        input: ndarray
        output: ndarray
        '''
        #first remove unused dims
        inputImg = inputImg.squeeze()
        nonZero = np.ma.masked_equal(inputImg,0)
        image = (nonZero*self.normalization["vars"]+self.normalization["means"]).data
        
        return image

    def standardize_nii(self,niiObject,minHU=-1000, maxHU=3000):
        '''
        Pre-process images by clipping outliers and shifting -1000 to 0
        Input: nibabel object (512,512,num_slices)
        Output: normalized ndarray  
        '''
        #1. convert nii object to ndarray
        img = niiObject.get_data()

        #2. standardize dimension from [H,W,N] to [N,H,W] 
        img = np.transpose(img,[2,0,1])

        #3. nii already converted to HU's
        imgClip = (img.clip(minHU,maxHU)+1000).astype('uint16')

        #4. save original image
        self.images = imgClip

        #5. images are rotated vertically correct them here
        imgRot = np.rot90(imgClip,k=3,axes=(1,2)); 

        return imgRot

    def de_standardize_nii(self,imgInput):
        '''
        Method to convert nii images back to original format
        inputs: ndarray
        output: ndarray 
        '''
        #1. rotate back image as above
        imgRot = np.rot90(imgInput,k=-3,axes=(1,2))    
        #2. convert [N,H,W] -> [H,W,N]
        #imgRot = np.transpose(imgRot,[1,2,0])
        
        return imgRot
    
    def standardize_nii_label(self,label):
        '''
        Pre-process labels by rotating horizontally    
        Input: niibabel object (512,512,num_slices)
        Output: standardized ndimage 
        '''
        #1. convert nii to ndarray
        label = label.get_data()

        #2. standardiza dimension from [H,W,N] to [N,H,W] 
        label = np.transpose(label,[2,0,1])

        #3. save original labels
        self.labels = label

        #4. rotate label counter-clockwise
        labelRot = np.rot90(label,k=3,axes=(1,2)) 

        return labelRot

    def standardize_dicom(self,imgSlice,minHU=-1000, maxHU=3000):
        '''
        Converts pixel values to CT values, then shifts everything by 1000 to make air zero
        Input: pydicom object
        Outpur: ndarray    
        '''
        #1. convert pixel data to HU
        slope = imgSlice.RescaleSlope
        intercept = imgSlice.RescaleIntercept
        imgClip = imgSlice.pixel_array*slope+intercept

        #print("Before clipping ",sliceHU.max(), sliceHU.min(), sliceHU.dtype);
        #2 clip HU between [-1000, 3000]
        imgClip = (imgClip.clip(minHU,maxHU)+1000).astype('float32')

        #3. save original image
        self.images = imgClip

        return imgClip

    def standardize_dicom_label(self,labelMask,removeBody=False, organToSegment=False):
        '''
        Method to standardize labels; such as removing body label or binarizing labels
        '''
        if organToSegment:
            labelMask[labelMask!=organToSegment] = 0;
            labelMask[labelMask==organToSegment] = 1;

        if removeBody:
            #1. find non zero mask
            nonAir = ~(labelMask==0);    
            #2. subtract one from index, 
            labelMask[nonAir] = labelMask[nonAir]-1; 
    
        #3. save original labels
        self.labels = labelMask

        return labelMask.astype("float32")

class Cropper:
    def __init__(self):
        self.rows = None
        self.cols = None

    def crop(self,imgInput):
        '''
        Method to crop image
        input: ndarray
        returns: list of images
        '''
        #first check dimensions of image
        if len(imgInput.shape)==2:
            rows, cols = body_bounding_box(imgInput)
            #update attributes            
            self.rows = [rows]
            self.cols = [cols]

            #crop image 
            return [crop_image_roi(imgInput,rowMin=rows[0],rowMax=rows[1],colMin=cols[0],colMax=cols[1])];

        elif len(imgInput.shape)==3:
            rows, cols , cropImg = [],[],[]
            N = imgInput.shape[0]
            
            for idx in range(N):
                row,col = body_bounding_box(imgInput[idx])
                rows.append(row)
                cols.append(col)
                #assign values to zoom
                tmp = crop_image_roi(imgInput[idx],rowMin=row[0],rowMax=row[1],colMin=col[0],colMax=col[1])
                cropImg.append(tmp)

            #update atributes
            self.rows = rows
            self.cols = cols
            
            #return cropped image
            return cropImg
        else:
            sys.exit("Error encountered in BoundingBoxCropper.bounding_box_util!")  
    
    def crop_label(self,labelInput):
        '''
        Method to crop center of label using self.cols and self.rows locations
        labelInput: label ndarray with (N,H,W,C) format
        output: list of cropped labels
        '''
        N = labelInput.shape[0]
        cropLabel = []
        for idx in range(N):
            row = self.rows[idx]
            col = self.cols[idx]
            #assign values to zoom
            tmp = crop_image_roi(labelInput[idx],rowMin=row[0],rowMax=row[1],colMin=col[0],colMax=col[1])
            cropLabel.append(tmp)
        return cropLabel

        
    def uncrop_image(self,row,col,imgInput):
        '''
        Method to pad cropped image with zeros
        input: 2D array
        output: 2D array
        '''
        rowPad = (row[0],H0-row[1])
        colPad = (col[0],W0-col[1])
        padImg = np.pad(imgInput,pad_width=(rowPad,colPad),mode='constant'); 
        return padImg

    def uncrop(self,imgInput):
        '''
        Method to pad image with zeros to return 512x512 resolution
        input: list (images have non-uniform shape so you can't concatenate)
        output: ndarray
        '''
        n = len(imgInput)
        unCropImg = np.zeros((n,H0,W0))
        for idx in range(n):
            col = self.cols[idx]
            row = self.rows[idx]
            unCropImg[idx] = self.uncrop_image(row,col,imgInput[idx])
            
        return unCropImg

    def zoom_image(self,imgInput,height=W,width=H):
        '''
        Method to zoom image, using Lancsoz interpolation over 8x8 pixel neighborhood. default: liner was not good
        input: 2d array
        returns: 2d array
        '''          
        zoomImg = cv2.resize(imgInput,(width,height),interpolation=cv2.INTER_LANCZOS4)
        return zoomImg

    def unzoom_image(self,row,col,imgInput):
        '''
        Method to dezoom image
        input: 2d array
        returns: 2d array
        '''
        deZoomImg = cv2.resize(imgInput.astype('float32'),(col[1]-col[0],row[1]-row[0]),interpolation=cv2.INTER_LANCZOS4)
        return deZoomImg

    def unzoom(self,zoomImg):
        '''
        Method to dezoom images in current batch
        To avoid ambuigity images of the current batch will be automatically used    
        '''
        unZoomImg = []
        for idx in range(zoomImg.shape[0]):
            col = self.cols[idx]
            row = self.rows[idx]
            tmp = self.unzoom_image(row,col,zoomImg[idx])
            unZoomImg.append(tmp)
        return unZoomImg 

    def unzoom_label(self,zoomLabel):
        '''
        Method to unzoom labels. Input is 4D array, so I need to zoom each label 
        separately. Then combine them. Can we optimize thi method???
        '''
        unZoomLabel = []
        for idx in range(zoomLabel.shape[0]):
            col = self.cols[idx]
            row = self.rows[idx]
            currLabel = np.zeros((row[1]-row[0],col[1]-col[0],zoomLabel.shape[-1]))
            for label in range(NUMCLASSES):
                #currLabel = cv2.resize(zoomLabel[idx],(col[1]-col[0],row[1]-row[0]),interpolation=cv2.INTER_NEAREST)
                tmp = cv2.resize(zoomLabel[idx,...,label],(col[1]-col[0],row[1]-row[0]),interpolation=cv2.INTER_LANCZOS4)
                currLabel[...,label] = tmp
            unZoomLabel.append(currLabel)
        return unZoomLabel

    def uncrop_label(self,cropLabel):
        '''
        Method to padd labels. Input is 4D array, so I need to pad each label 
        separately. Then combine them. Can we optimize thi method???
        Input: list()
        Output: 4D array
        '''
        unCropLabel = np.zeros((len(cropLabel),H0,W0,NUMCLASSES))
        for idx in range(len(cropLabel)):
            col = self.cols[idx]
            row = self.rows[idx]
            currLabel = np.zeros((H0,W0,NUMCLASSES))
            for label in range(NUMCLASSES):
                tmp = self.uncrop_image(row,col,cropLabel[idx][...,label])
                currLabel[...,label] = tmp
            unCropLabel[idx] = currLabel
        return unCropLabel

    def zoom_label(self,labelInput,height=W,width=H):
        '''
        Method to zoom batch of labels. 
        input: can be list or ndarray
        returns: 4D array (N,H,W,C)
        '''
        if type(labelInput)==np.ndarray:
            n = labelInput.shape[0]
        elif type(labelInput)==list:
            n = len(labelInput)
        else:
            sys.exit(type(self).__name__+'.zoom_label() accepts list or nd array'+type(labelInput)+' provided');

        zoomedLabel = np.zeros((n,height,width,NUMCLASSES))
        #cropImg is list so, need to iterate 
        for idx in range(n):
            #col = self.cols[idx]
            #row = self.rows[idx]
            row,col = labelInput[idx].shape

            #convert each image to categorical
            labelOneHot = to_categorical(labelInput[idx],num_classes=NUMCLASSES).reshape((row,col,NUMCLASSES))
            #labelOneHot = to_categorical(labelInput[idx],num_classes=NUMCLASSES).reshape((row[1]-row[0],col[1]-col[0],NUMCLASSES))
            for label in range(NUMCLASSES):
                zoomedLabel[idx,...,label] = cv2.resize(labelOneHot[...,label],(width,height),interpolation=cv2.INTER_LANCZOS4)

        return zoomedLabel

    def zoom(self,imgInput,height=W,width=H):
        '''
        Method to zoom batch of images
        input: can be list or ndarray
        returns: ndarray
        '''
        if type(imgInput)==np.ndarray:
            n = imgInput.shape[0]
        elif type(imgInput)==list:
            n = len(imgInput)
        else:
            sys.exit(type(self).__name__+'.zoom() accepts list or nd array'+type(imgInput)+' provided');

        zoomedImg = np.zeros((n,height,width))
        #cropImg is list so, need to iterate 
        for idx in range(n):
            tmp = self.zoom_image(imgInput[idx],height,width)
            zoomedImg[idx] = tmp; 
        
        return zoomedImg

class ImageProcessor(Normalizer,Cropper):
	
    def __init__(self,normalize=None):
        Normalizer.__init__(self,normalize)
        Cropper.__init__(self)

    def standardize_img(self,inputFile):
        #first convert image to numpy array
        if type(inputFile)==nib.nifti1.Nifti1Image:#input file is nii
            imgStandard = self.standardize_nii(inputFile)
        elif type(inputFile)==pydicom.dataset.FileDataset:#input file is dcm
            imgStandard = self.standardize_dicom(inputFile)
        elif type(inputFile)==np.ndarray:#input must be already standardized
            self.images = inputFile
            imgStandard = inputFile
        else:
            sys.exit(type(self).__name__+".standardize_img can't standardize inpuy file")
        return imgStandard

    def standardize_label(self,inputFile):
        #first convert image to numpy array
        if type(inputFile)==nib.nifti1.Nifti1Image:#input file is nii
            labelStandard = self.standardize_nii_label(inputFile)
        elif type(inputFile)==pydicom.dataset.FileDataset:#input file is dcm
            labelStandard = self.standardize_dicom_label(inputFile)
        elif type(inputFile)==np.ndarray:#input must be already standardized
            self.labels = inputFile
            labelStandard = inputFile
        else:
            sys.exit(type(self).__name__+".standardize_label can't standardize inpuy file")
        return labelStandard

    def img_to_tensor(self,array):
        '''
        Method to convert numpy array to 4D tensor to process in tensorflow
        Input: ndarray, must be at least 2D image
        Output: 4D ndarray
        '''
        #1. get array shape
        shape = array.shape

        if len(shape)==2:
            return array[np.newaxis,...,np.newaxis]
        elif len(shape)==3:
            #num channel exist, batch size missing
            if (np.prod(shape[1:])==H0*W0) or (np.prod(shape[1:])==H*W):
                return array[...,np.newaxis]
            #num batches exist but, channel missing
            elif (np.prod(shape[:-1])==H0*W0) or (np.prod(shape[:-1])==H*W):        
                return array[np.newaxis,...]
            else:
                sys.exit("preprocessing.img_to_tensor method can't convert ",shape," to 4D tensor");
        elif len(shape)==4:#already 4D tensor
            return array
        else:
            sys.exit("preprocessing.img_to_tensor method can't convert ",shape," to 4D tensor");

    def pre_process_img_label(self,imgBatch,labelBatch,crop=True):
        '''
        Method to pre-process image and label simultanously. 
        imgBatch: standardized ndarray (CT number clipped, transposed and rotated)
        labelBatch: pre-processed label (enumerated label, transposed and rotated)
        '''
        #1. pre-processing image only
        imgNorm = self.pre_process_img(imgBatch,crop)
        #2. pre-process label
        labelZoom = self.pre_process_label(labelBatch,crop)

        return imgNorm, labelZoom

    def pre_process_label(self,labelInput,crop=True):
        '''
        Method to pre-process label file. Zooming slightly distorts label mask, it seems effect is small (<3%)
        inputFile: .nii, dcm or ndarray
        output: cropped label mask, shape=(N,H,W,C)
        ''' 
        if crop:
            labelStandard = self.standardize_label(labelInput)
            labelCrop = self.crop_label(labelStandard)
            labelZoom = self.zoom_label(labelCrop)
            labelZoomCat = to_categorical((labelZoom).argmax(axis=-1),NUMCLASSES).reshape((-1,W,H,NUMCLASSES))
        else:#croping already applied to just run on full resolution
            labelStandard = self.standardize_label(labelInput)
            labelZoomCat = to_categorical(labelStandard,NUMCLASSES).reshape((-1,W,H,NUMCLASSES))
        return labelZoomCat

    def pre_process_img(self,inputFile,crop=True):
        '''
        Method to pre-process input file
        inputFile: .nii or dcm file
        output: cropped and normalized ndarray, which can be directly passed to model
        ''' 
        if crop:
            imgStandard = self.standardize_img(inputFile) #convert to numpy
            imgCrop = self.crop(imgStandard)
            imgZoom = self.zoom(imgCrop)
            imgNorm = self.normalize(imgZoom)
        else: #normalization is 256x384 so input has to be cropped
            imgStandard = self.standardize_img(inputFile)
            imgNorm = self.normalize(imgStandard)
        return imgNorm

    def inverse_pre_process_img(self,imgNorm):
        #apply inverse preprocessing
        imgDeNorm = self.denormalize(imgNorm)
        imgUnZoom = self.unzoom(imgDeNorm)
        imgDeCrop = self.uncrop(imgUnZoom)

        return imgDeCrop

    def morphological_operation(self,img,operation='close'):
        '''
        post-processing methods on label
        TODO:fix this
        '''
        kernel = np.ones((5,5),np.uint8);

        if len(img.shape)==2:
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        else:
            outImg = np.zeros_like(img);
            for idx in range(img.shape[-1]):
                if operation=='close':
                    outImg[...,idx] = cv2.morphologyEx(img[...,idx], cv2.MORPH_CLOSE, kernel)
                elif operation=='open':
                    outImg[...,idx] = cv2.morphologyEx(img[...,idx], cv2.MORPH_OPEN, kernel)
                elif operation=='dilate':
                    outImg[...,idx] = cv2.morphologyEx(img[...,idx], cv2.MORPH_DILATE, kernel)
                else:
                    sys.exit('morphological operation invalid ')
            return outImg

    def morphological_operation_3d(self,img,operation='close'):
        '''
        apply morphological operation 3D 
        '''
        # kernel = np.ones((5,5),np.float32)
        kernel = np.ones((3,3),np.float32)
        # pdb.set_trace()

        if len(img.shape)==2:
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        else:
            outImg = np.zeros_like(img)
            for idx in range(img.shape[-1]):
                #second apply on x-axis
                for x in range(img.shape[1]):
                    if operation=='close':
                        outImg[:,x,:,idx] = grey_closing(outImg[:,x,:,idx], structure=kernel)
                    elif operation=='open':
                        outImg[:,x,:,idx] = grey_opening(outImg[:,x,:,idx], structure=kernel)
                    elif operation=='dilate':
                        outImg[:,x,:,idx] = grey_dilation(img[:,x,:,idx], structure=kernel)
                    else:
                        sys.exit('morphological operation invalid ')
                #first apply on y-axis
                for y in range(img.shape[2]):
                    if operation=='close':
                        outImg[:,:,y,idx] = grey_closing(img[:,:,y,idx], structure=kernel)
                    elif operation=='open':
                        outImg[:,:,y,idx] = grey_opening(img[:,:,y,idx], structure=kernel)
                    elif operation=='dilate':
                        outImg[:,:,y,idx] = grey_dilation(img[:,:,y,idx], structure=kernel)
            return outImg
    
from scipy.ndimage.morphology import *

def body_bounding_box(img):
    '''
    Method to automatically create bounding-box around the body of the ct slice. Here is cropping is done
        1. Find mass center of the image in rx, ry axis
        2. Binarize image using 200 threshold
        3. Apply morpholical opening to the top half of image
            Apply morphological eroding to the bottom half of the image
        4. calculate first non-zero location in both directions         

    Input: 2D ndarray
    Output: tuple of coordinates (rowMin, rowMax), (colMin, colMax)    
    '''

    #define kernel
    SIZE1 = 11; SIZE2 = 21;    
    kernel1 = np.ones((SIZE1,SIZE1),np.uint8)
    kernel2 = np.ones((SIZE2,SIZE2),np.uint8)

    #calculate center points
    row_mean = img.mean(axis=1)
    rows = np.arange(img.shape[0])
    row_center = np.int((rows*row_mean).sum()//row_mean.sum())
    # col_mean = img.mean(axis=0)
    # cols = np.arange(img.shape[1])
    # col_center = np.int((cols*col_mean).sum()//col_mean.sum())

    #create label img
    label = np.zeros((W0,H0),dtype=np.uint8)
    body = label.copy()

    #first binarize image
    label[img>600] = 1

    #add white spot at the center
    #img[row_center-5:row_center+5,col_center-5:col_center+5] = img.max();

    #apply morphology
    body[:row_center] = cv2.morphologyEx(label[:row_center], cv2.MORPH_OPEN, kernel1)
    body[row_center:] = cv2.morphologyEx(label[row_center:], cv2.MORPH_ERODE, kernel2)

    #compute non zeros
    rows = np.any(body, axis=1)
    cols = np.any(body, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    #apply margin
    ymin, ymax = max(0,ymin-SIZE1//2), min(ymax+SIZE2//2,512)
    xmin, xmax = max(0,xmin-SIZE1//2), min(xmax+SIZE1//2,512) 
    
    #return crop positions
    return (ymin,ymax),(xmin,xmax)

def crop_image_roi(img,rowMin=ROW,rowMax=ROW+W,colMin=COL,colMax=COL+H):
    '''
    Method to crop center of image using pre-defined regions. Not using it anymore
    Input: can be 2D, 3D (may have bug, assumes [N,H,W]) or 4D image
    Output: 
    '''
    if len(img.shape)==2:
        return img[rowMin:rowMax,colMin:colMax]
    elif len(img.shape)==3:
        return img[:,rowMin:rowMax,colMin:colMax]
    elif len(img.shape)==4:
        return img[:,rowMin:rowMax,colMin:colMax,:]
    else:
        sys.exit("preprocessing.crop_image_roi method can't crop img size of",img.shape)
        



