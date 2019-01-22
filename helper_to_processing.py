import os, sys, glob
import numpy as np
import nibabel as nib
import pydicom

import cv2
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

from config import *


class Normalizer:
    def __init__(self,normalize=None):
        self.normalization = normalize;

    def normalize(self,inputImg):
        nonZero = np.ma.masked_equal(inputImg,0)
        normalized = ((nonZero-self.normalization["means"])/self.normalization["vars"]).data
        return normalized

    def denormalize(self,inputImg):
        nonZero = np.ma.masked_equal(inputImg,0)
        image = (nonZero*self.normalization["vars"]+self.normalization["means"]).data
        return image

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
        Method to padd image with zeros to return 512x512 resolution
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
        returns: ndarray
        '''          
        zoomImg = cv2.resize(imgInput,(width,height),interpolation=cv2.INTER_LANCZOS4)
        return zoomImg

    def unzoom_image(self,row,col,imgInput):
        '''
        Method to dezoom image
        input: 2d array
        returns: ndarray
        '''
        deZoomImg = cv2.resize(imgInput,(col[1]-col[0],row[1]-row[0]),interpolation=cv2.INTER_LANCZOS4)
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

    def zoom(self,imgInput,height=W,width=H):
        '''
        Method to crop and zoom image
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
            imgStandard = standardize_nii(inputFile)
        elif type(inputFile)==pydicom.dataset.FileDataset:#input file is dcm
            imgStandard = standardize_slice(inputFile)
        elif type(inputFile)==np.ndarray:#input must be already standardized
            imgStandard = inputFile
        else:
            sys.exit(type(self).__name__+".pre_process_img can't standardize inpuy file")
        return imgStandard

    def pre_process(self,inputFile):
        #preprocessing 
        imgStandard = self.standardize_img(inputFile)
        imgCrop = self.crop(imgStandard)
        imgZoom = self.zoom(imgCrop)
        imgNorm = self.normalize(imgZoom)
        
        return imgNorm

    def inverse_pre_process(self,imgNorm):
        #apply inverse preprocessing
        imgDeNorm = self.denormalize(imgNorm)
        imgUnZoom = self.unzoom(imgDeNorm)
        imgDeCrop = self.uncrop(imgUnZoom)

        return imgDeNorm

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


def crop_body_roi(imgInput,labelInput):
    '''
    Wrapper function to crop batches of image and label pair. If label is not provided image is cropped, otherwise both are cropped. 
    Input: ndarray; img or img/label pair
    Output: ndarray; img or img/label pair
    '''

    #remove axis
    img = imgInput.squeeze();

    if labelInput is not None:
        #first remove body mask from labels, for nii images there was not body contours
        #remove_body_mask(labelInput);
        #convert to one-hot encoded
        label = to_categorical(labelInput,num_classes=NUMCLASSES).reshape((-1,W0,H0,NUMCLASSES));   

    #first check dimensions of image
    if len(img.shape)==2:
        rows, cols = body_bounding_box(imgInput);
        #crop and resize image
        imgCrop = imgInput[rows[0]:rows[1],cols[0]:cols[1]];
        imgZoom = cv2.resize(imgCrop,(H,W));

        if labelInput is not None:
            #crop and resize label
            labelCrop = label[rows[0]:rows[1],cols[0]:cols[1],:];        
            labelZoom = resize(labelCrop,(W,H,NUMCLASSES));
            return imgZoom, labelZoom
        else:
            return imgZoom

    elif len(img.shape)==3:
        N = img.shape[0];
        cropImg = np.zeros((N,W,H));
        cropLabel = np.zeros((N,W,H,NUMCLASSES));

        for idx in range(N):
            rows, cols = body_bounding_box(imgInput[idx]);
            #crop and resize image
            imgCrop = imgInput[idx,rows[0]:rows[1],cols[0]:cols[1]];
            imgZoom = cv2.resize(imgCrop,(H,W));
            cropImg[idx] = imgZoom;
        
            if labelInput is not None:
                #crop and resize label
                labelCrop = label[idx,rows[0]:rows[1],cols[0]:cols[1],:];        
                labelZoom = resize(labelCrop,(W,H,NUMCLASSES));
                cropLabel[idx] = labelZoom;

        if labelInput is not None:
            return cropImg, cropLabel
        else:
            return cropImg
    else:
        sys.exit("preprocessing.crop_body_roi can't crop img size"+str(img.shape)+"; Input must be (H,W) or (N,H,W)")


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
        


def standardize_nii(img,minHU=-1000, maxHU=3000):
    '''
    Pre-process images by clipping outliers and shifting -1000 to 0
    Input: nibabel object (512,512,num_slices)
    Output: normalized ndarray  
    '''
    #1. convert nii object to ndarray
    img = img.get_data()

    #2. standardiza dimension from [H,W,N] to [N,H,W] 
    img = np.transpose(img,[2,0,1])

    #print("CT before clipping ...",img.max(), img.min())
    #2. nii already converted to HU's
    sliceHU = (img.clip(minHU,maxHU)+1000).astype('uint16')
    #print("CT after clipping ...",sliceHU.max(), sliceHU.min())

    #3. images are rotated vertically correct them here
    imgRot = np.rot90(sliceHU,k=3,axes=(1,2)); 

    return imgRot

def standardize_nii_label(label):
    '''
    Pre-process labels by rotating horizontally    
    Input: niibabel object (512,512,num_slices)
    Output: standardized ndimage 
    '''
    #1. convert nii to ndarray
    label = label.get_data()

    #3. standardiza dimension from [H,W,N] to [N,H,W] 
    label = np.transpose(label,[2,0,1])

    #2. roate label counter-clockwise
    labelRot = np.rot90(label,k=3,axes=(1,2)) 

    return labelRot

def standardize_slice(imgSlice,minHU=-1000, maxHU=3000):
    '''
    Converts pixel values to CT values, then shifts everything by 1000 to make air zero
    Input: pydicom object
    Outpur: ndarray    
    '''
    #1. convert pixel data to HU
    slope = imgSlice.RescaleSlope
    intercept = imgSlice.RescaleIntercept
    sliceHU = imgSlice.pixel_array*slope+intercept

    #print("Before clipping ",sliceHU.max(), sliceHU.min(), sliceHU.dtype);
    #2 clip HU between [-1000, 3000]
    sliceHU = (sliceHU.clip(minHU,maxHU)+1000).astype('uint16')
    
    return sliceHU

def remove_body_mask(labelMask):
    '''
    Modifies labels in place (be-carefull using this method).
    Body labels are marked as one, this method that to zero and decrements other labels by one    
    Input: ndarray
    Outpur: ndarray, modifiesarray in place
    '''
    #1. find non zero mask
    nonAir = ~(labelMask==0);    
    #2. subtract one from index, 
    labelMask[nonAir] = labelMask[nonAir]-1;  

def img_to_tensor(array):
    '''
    Method to convert numpy array to 4D tensor to process in tensorflow
    Input: ndarray, must be at least 2D image
    Output: 4D ndarray
    '''
    #1. get array shape
    shape = array.shape;

    if len(shape)==2:
        return array[np.newaxis,...,np.newaxis];
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


def pre_process_img_label(imgInput,labelInput=None,normalize=None):
    '''
    Wrapper method to pre-process and crop batches of img/label pairs.
    Input: ndarray img or img/label pair
    Output: ndarray img or img/label pair
    '''

    #crop label and image
    if labelInput is not None:
        cropImg, cropLabel = crop_body_roi(imgInput,labelInput);
    else:
        cropImg = crop_body_roi(imgInput,labelInput);

    #apply normalization to image
    if normalize is not None:
        nonZero = np.ma.masked_equal(cropImg,0);
        normalized = ((nonZero-normalize["means"])/normalize["vars"]).data;
    else:
        normalized = cropImg;    

    if labelInput is not None:
        #return processed image label pair
        return normalized.astype("float32"), cropLabel.astype("float32");
    else:
        return normalized.astype("float32")

def pre_process_img(imgInput,normalize,removeAir=True):
    """
    Normalizes image in ROI
    """
    if removeAir:
        nonZero = np.ma.masked_equal(imgInput,0);
        normalized = ((nonZero-normalize["means"])/normalize["vars"]).data;
    else:
        normalized = (imgInput-normalize["means"])/normalize["vars"];

    #crop image
    normalized = crop_image_roi(normalized);

    return normalized.astype("float32");

def pre_process_label(label,organToSegment=None):
    """
    Preprocesses label by removing body
    """
    #crop image for roi
    label = crop_image_roi(label);

    #remove body mask
    remove_body_mask(label)

    #Add preprocessing for labels, applies on training stage
    if organToSegment:
        label[label!=organToSegment] = 0;
        label[label==organToSegment] = 1;

    return label.astype("float32");
    

