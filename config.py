import numpy as np

########################
#### CONFIG ResNet #####
FILTER1 = 16;#24
FILTER2 = 32;#24
FILTER3 = 64;#32
FILTER4 = 128;#64
FILTER5 = 256;#128
FILTER6 = 512
FILTER7 = 1024;
BATCHSIZE = 8;
NUMEPOCHS = 100;
NUMCLASSES = 5;
L2PENALTY = 0.0001;
LEARNRATE = 0.0001#0.0001
TRAINSIZE, VALSIZE = 6396, 1024;
STEPPEREPOCHS = int(TRAINSIZE/BATCHSIZE); 
VALSTEPS = int(VALSIZE/BATCHSIZE); 
DECAYRATE = 1/(STEPPEREPOCHS*NUMEPOCHS);
#CLASSWEIGHTS = {0: 1.0, 1: 2148, 2: 128, 3: 2864, 4: 525} #calculated globally (total_for_all_categories/total_for_category)
CLASSWEIGHTS = np.array([ 1., 7.67229246,  4.85203026,  7.95997453,  6.26339826]); #logarithm of above numbers

#image crop indeces
ROW, COL = 115,54
H,W,C = 384,256,1
H0,W0,C0 = 512,512,1

#using default model with loss categorical-loss function, no data stratifying
#modelName = "1x256x384_Base_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

#training with weighted loss function
#modelName = "1x256x384_WeightedLoss_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

#training by augmenting data
#modelName = "1x256x384_Augmented_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

#same model as before but multiclass loss function
modelName = "1x256x384_MultiClass_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

########################

