#Dependecias de Django
from django.shortcuts import render, redirect
from apps.Archivo.models import CSV
import math
from tqdm.auto import tqdm
from django.views.generic import ListView,CreateView,UpdateView,DeleteView,TemplateView
from django.urls import reverse_lazy
from django.contrib import messages
from apps.Diagnostico.forms import DiagnosticoForm

import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image


from apps.ImagenMRI.models import ImagenMRI
from apps.Sujeto.models import Sujeto
from apps.ImagenMascara.models import ImagenMascara
from apps.Overlay.models import Overlay
from apps.TablaCaracteristicas.models import TablaCaracteristicas
from apps.Diagnostico.models import Diagnostico
from apps.Doctor.models import Doctor

#Dependecias utiles para logica de nuestro modelo de clasificacion
import csv
import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from pandas import read_csv,DataFrame,read_excel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.externals import joblib
import joblib
#Dependencias utiles para logica de modelo de segmentacion U-net
import os
import sys
import time
import warnings
import scipy
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation, Dense, Flatten
from keras.activations import elu
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import dicom2nifti
import dicom2nifti.settings as settings
#from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
from PIL import Image
from resizeimage import resizeimage
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential, Model
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import VGG19
import med2image
import SimpleITK as sitk
K.set_image_data_format('channels_last')

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
smooth=1.
global a
model_path = 'media/models/_85_15_unet2.h5'
Path_MASK_folder1 =  'media/ImagenesMascarasBinarias/'
Path_MASK_folder_Int =  'media/ImagenesMascaras/'
# Create your views here.

#def mask_X_size(file):
#    image_path = sitk.ReadImage(file)
#    image_path_array = sitk.GetArrayFromImage(image_path)
#    return image_path_array.shape[0]

def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)

def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    #print("aqui muestra canales o capas")
    #print(cw)
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

#post procesamiento Flair generacion de mascara
def general_postprocessing(FLAIR_array, pred):
    start_slice = 6
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0

    if (image_rows_Dataset >= rows_standard and image_cols_Dataset >= cols_standard):
        original_pred[:,int((image_rows_Dataset-rows_standard)/2):int((image_rows_Dataset+rows_standard)/2),int((image_cols_Dataset-cols_standard)/2):int((image_cols_Dataset+cols_standard)/2)] = pred[:,:,:,0]
        
        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred
    elif (image_rows_Dataset >= rows_standard and image_cols_Dataset < cols_standard):
        original_pred[:, int((image_rows_Dataset-rows_standard)/2):int((image_rows_Dataset+rows_standard)/2),:] = pred[:,:, int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2),0]

        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred


    elif (image_rows_Dataset < rows_standard and image_cols_Dataset >= cols_standard):
        original_pred[:, :,int((image_cols_Dataset-cols_standard)/2):int((image_cols_Dataset+cols_standard)/2)] = pred[:,int((rows_standard-image_rows_Dataset)/2):int((rows_standard+image_rows_Dataset)/2),:,0]

        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred

    else:
        original_pred = pred[:,int((rows_standard-image_rows_Dataset)/2):int((rows_standard+image_rows_Dataset)/2),int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2),0]

        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred

#post procesamiento FLAIR y T1
def general_preprocessing(FLAIR_image, T1_image):
    channel_num = 2
    #print(np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    T1_image = np.float32(T1_image)

    if (image_rows_Dataset >= rows_standard and image_cols_Dataset >= cols_standard):
        brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
        imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)

        # FLAIR --------------------------------------------
        brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
        brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
        for iii in range(np.shape(FLAIR_image)[0]):
            brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
        FLAIR_image = FLAIR_image[:, (int(image_rows_Dataset/2-rows_standard/2)):(int(image_rows_Dataset/2+rows_standard/2)), (int(image_cols_Dataset/2-cols_standard/2)):(int(image_cols_Dataset/2+cols_standard/2))]
        brain_mask_FLAIR = brain_mask_FLAIR[:, (int(image_rows_Dataset/2-rows_standard/2)):(int(image_rows_Dataset/2+rows_standard/2)), (int(image_cols_Dataset/2-cols_standard/2)):(int(image_cols_Dataset/2+cols_standard/2))]
        ###------Gaussion Normalization here
        np.subtract(FLAIR_image, np.mean(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
        np.divide(FLAIR_image, np.std(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        T1_image = T1_image[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
        brain_mask_T1 = brain_mask_T1[:,int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2),int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
        #------Gaussion Normalization
        np.subtract(T1_image, np.mean(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
        np.divide(T1_image, np.std(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image /=np.std(T1_image[brain_mask_T1 == 1])
        #---------------------------------------------------
        FLAIR_image  = FLAIR_image[..., np.newaxis]
        T1_image  = T1_image[..., np.newaxis]
        imgs_two_channels = np.concatenate((FLAIR_image, T1_image), axis = 3)
        return imgs_two_channels
    elif (image_rows_Dataset >= rows_standard and image_cols_Dataset < cols_standard):

        brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
        imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
        FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
        T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
        
        # FLAIR --------------------------------------------
        brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
        brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
        for iii in range(np.shape(FLAIR_image)[0]):
    
            brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        np.subtract(FLAIR_image, np.mean(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
        np.divide(FLAIR_image, np.std(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
        FLAIR_image_suitable[...] = np.min(FLAIR_image)
        FLAIR_image_suitable[:, :,int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2)] = FLAIR_image[:,int (image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), :]
    
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
    
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        np.subtract(T1_image, np.mean(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
        np.divide(T1_image, np.std(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:, :, int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2)] = T1_image[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), :]
        #---------------------------------------------------
        FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
        T1_image_suitable  = T1_image_suitable[..., np.newaxis]
        
        imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
        return imgs_two_channels
    elif (image_rows_Dataset < rows_standard and image_cols_Dataset >= cols_standard):
        brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
        imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
        FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
        T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

        # FLAIR --------------------------------------------
        brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
        brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
        for iii in range(np.shape(FLAIR_image)[0]):
    
            brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        np.subtract(FLAIR_image, np.mean(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
        np.divide(FLAIR_image, np.std(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
        FLAIR_image_suitable[...] = np.min(FLAIR_image)
        FLAIR_image_suitable[:, int((rows_standard - image_rows_Dataset)/2):int((rows_standard + image_rows_Dataset)/2),:] = FLAIR_image[:, :, int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
    
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        np.subtract(T1_image, np.mean(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
        np.divide(T1_image, np.std(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:,int((rows_standard - image_rows_Dataset)/2):int((rows_standard + image_rows_Dataset)/2),:] = T1_image[:, :, int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
        #---------------------------------------------------
        FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
        T1_image_suitable  = T1_image_suitable[..., np.newaxis]
        
        imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
        return imgs_two_channels
    else:
        brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
        imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
        imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
        FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
        T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

        # FLAIR --------------------------------------------
        brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
        brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
        for iii in range(np.shape(FLAIR_image)[0]):
    
            brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        np.subtract(FLAIR_image, np.mean(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
        np.divide(FLAIR_image, np.std(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
        FLAIR_image_suitable[...] = np.min(FLAIR_image)
        FLAIR_image_suitable[:, int((rows_standard - image_rows_Dataset)/2):int((rows_standard + image_rows_Dataset)/2),int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2)] = FLAIR_image[...]
    
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
    
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        np.subtract(T1_image, np.mean(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
        np.divide(T1_image, np.std(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:, int((rows_standard - image_rows_Dataset)/2):int((rows_standard + image_rows_Dataset)/2),int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2)] = T1_image[...]
        #---------------------------------------------------
        FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
        T1_image_suitable  = T1_image_suitable[..., np.newaxis]
        
        imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
        return imgs_two_channels

# definición del modelo para validacion del cerebro
def define_model():
    # cargar modelo
    model = VGG19(include_top=False, input_shape=( 192,256, 3))
    # indicar a las capas no entrenar
    for layer in model.layers:
        layer.trainable = False
    # añadir mi clasificador: una nn de 1 capa full con 128 neuronas
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # definición del nuevo modelo
    model = Model(inputs=model.inputs, outputs=output)
    # compilar modelo
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# definición del modelo para segmentacion de WMH
def get_unet_2(img_shape = None, first5=True):
    inputs = Input(shape = img_shape)
    concat_axis = -1

    if first5: filters = 5
    else: filters = 3
    conv1 = conv_bn_relu(64, filters, inputs)
    conv1 = conv_bn_relu(64, filters, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv3 = conv_bn_relu(128, 3, pool1)
    conv3 = conv_bn_relu(128, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(256, 3, pool3)
    conv4 = conv_bn_relu(256, 4, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(512, 3, pool4)
    conv5 = conv_bn_relu(512, 3, conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = conv_bn_relu(256, 3, up6)
    conv6 = conv_bn_relu(256, 3, conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = conv_bn_relu(128, 3, up7)
    conv7 = conv_bn_relu(128, 3, conv7)

    up_conv8 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = conv_bn_relu(64, 3, up9)
    conv9 = conv_bn_relu(64, 3, conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

    return model

#extraer caracteristicas usando funcion que etiqueta y captura objetos por su forma o intesidad
#tiene caracter estadistico(intensidad)
def extraerCaracteristicas(Final_MASK_image_path,name):
    #Lectura de las mascara- con intensidades
    Mask_image = sitk.ReadImage(Final_MASK_image_path,sitk.sitkInt32)
    #Declaramos las dos variables para obtener información segun forma o intensidad
    #Intensidad
    cc = sitk.ConnectedComponent(Mask_image>0)
    intensity_stats = sitk.LabelIntensityStatisticsImageFilter()
    intensity_stats.Execute(cc,Mask_image)
    #Forma
    cc = sitk.ConnectedComponentImageFilter()
    cca_image=cc.Execute(Mask_image)
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(cca_image)

    nombrePaciente=name

    stats_list = [(round(intensity_stats.GetKurtosis(i), 2),
                #intensity_stats.GetPhysicalSize(i),
                  round(intensity_stats.GetRoundness(i), 2),
                  round(intensity_stats.GetPerimeter(i), 2),
                  round(intensity_stats.GetVariance(i), 2),
                  round(intensity_stats.GetStandardDeviation(i), 2),
                #intensity_stats.GetElongation(i),
                    round(shape_stats.GetPhysicalSize(i), 2),
                    round(shape_stats.GetElongation(i), 2),
                    round(shape_stats.GetFlatness(i), 2)) for i in intensity_stats.GetLabels()]
    cols=["Curtosis_intensidad",
        #"Área_intensidad",
        "Redondez_intensidad",
        "Perimetro_intensidad",
        "Varianza_intensidad",
        #"Elongación_intensidad",
        "Desviación estandar_intensidad",
        "Volumen_forma (nm^3)",
        "Elongación_forma",
        "Flatness(llanura)_forma",]
    stats = pd.DataFrame(data=stats_list, index=intensity_stats.GetLabels(), columns=cols)
    #stats.describe()
    xpru=stats.mean()
    #print(xpru)
    stats_list = [ (
                              nombrePaciente,
                              round(xpru[0], 2),
                              round(xpru[1], 2),
                              round(xpru[2], 2),
                              round(xpru[3], 2),
                              round(xpru[4], 2),
                              round(xpru[5], 2),
                              round(xpru[6], 2),
                              round(xpru[7], 2),
                              intensity_stats.GetNumberOfLabels())]
    return stats_list

'''
Guarda una imagen nifty en jpg; si tiene un solo corte en z se lo toma caso contrario
se calcula la mitad del corte en z y luego es procesado como un solo slice
Ej: saveMRINiftytoJPG("./mr1.3.12.2.1107.5.2.32.35170.201102231753385275793382.0.0.0_t2_tirm_TRA_dark-fluid_3mm_20110223165745_9.nii.gz",".","mr1.3.12.2.1107.5.2.32.35170.201102231753385275793382.0.0.0_t2_tirm_TRA_dark-fluid_3mm_20110223165745_9")
'''
def saveMRINiftytoJPG(PathSource,PathFolderfinal,Finalname,format):
  if format=="dcm":
    '''itkimage = sitk.ReadImage(PathSource)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    print(numpyImage.shape)
    print(numpyImage.shape[0])
    slice_Number=numpyImage[0]-1
    im = Image.fromarray(numpyImage[0,:,: ].astype(np.uint8)) 
    im=im.convert('RGB')
    im.save(PathFolderfinal+Finalname+".png")'''
    return -1
  else:
    my_img  = nib.load(PathSource)
    nii_data = my_img.get_fdata()
    z=nii_data.shape[2]
    n=len(nii_data.shape)
    if n==3:
      if z==1:
        im = Image.fromarray(nii_data[:,:,z-1 ].astype(np.uint8))
        im=im.convert('RGB')
        im.save(PathFolderfinal+Finalname+".png")
        return 1
        #os.system("python3 media/med2image.py -i "+str(PathSource)+" -d "+str(PathFolderfinal)+" -o "+str(Finalname)+ " --outputFileType jpg")
        #med2image -i PathSource -d PathFolderfinal -o Finalname --outputFileType jpg 
        #return 1
      elif z>1:
        z=int(z/2)
        im = Image.fromarray(nii_data[:,:,z-1].astype(np.uint8))
        im=im.convert('RGB')
        im.save(PathFolderfinal+Finalname+".png")
        return 1
      else:
        return -1
    elif n>3:
      '''for frame in range(nii_data.shape[3]):
        for slice_Number in range(nii_data.shape[2]):
          im = Image.fromarray(nii_data[:,:,slice_Number,frame].astype(np.uint8))
          im.save(PathFolderfinal+Finalname+"_"+slice_Number+".png")
          print("Sleep")'''
      print("oye estas ingresando un overley... ")
      return -2
'''def saveMRINiftytoJPG(PathSource,PathFolderfinal,Finalname):
  grid_image = sitk.ReadImage(PathSource)
  nda = sitk.GetArrayFromImage(grid_image)
  z=nda.shape[0]
  #os.system('dir media')
  if z==1:
    path = PathSource
    my_img  = nib.load(path)
    slice_Number=0
    nii_data = my_img.get_fdata()
    #nii_aff  = my_img.affine
    print(nii_data.shape)
    #nii_hdr  = my_img.header
    im = Image.fromarray(nii_data[:,:,slice_Number ].astype(np.uint8))
    im=im.convert('RGB')
    im.save(PathFolderfinal+Finalname+".png")
    #os.system("python3 media/med2image.py -i "+str(PathSource)+" -d "+str(PathFolderfinal)+" -o "+str(Finalname)+ " --outputFileType jpg")
    #med2image -i PathSource -d PathFolderfinal -o Finalname --outputFileType jpg 
    return 1
  elif z>1:
    n=int(z/2)

    path = PathSource
    my_img  = nib.load(path)
    slice_Number=n-1
    nii_data = my_img.get_fdata()
    #nii_aff  = my_img.affine
    print(nii_data.shape)
    #nii_hdr  = my_img.header
    im = Image.fromarray(nii_data[:,:,slice_Number ].astype(np.uint8))
    im=im.convert('RGB')
    im.save(PathFolderfinal+Finalname+".png")

    #print("AQUI ESTA n---------------------------->",n)
    #os.system("python3 media/med2image.py -i "+str(PathSource)+" -d "+str(PathFolderfinal)+" -o "+ str(Finalname)+ " --outputFileType jpg --sliceToConvert "+str(n))
    #med2image -i PathSource -d PathFolderfinal -o Finalname --outputFileType jpg --sliceToConvert n
    return 1
  else:
    return -1
'''


'''
Cambia la dimension a la imagen a un estandar (192, 256, 3)
Ej: changedim('0_output-slice000.jpg')
'''
def changedim(Img):
  with open(Img, 'r+b') as f:
      with Image.open(f) as image:
          cover = resizeimage.resize_cover(image, [256, 192])
          cover.save(Img, image.format)


'''
Funcion que retorna verdadero o falso para imagen del cerebro JPG usando un modelo predefinido
Ej: isBraimJPG("0_output-slice000.jpg","brain_not_brain.h5")
'''
def isBraimJPG(Img,modelo):
  # load an image from file
  #image = load_img('rY99.jpg' ,target_size=(256, 192,3))
  image = load_img(Img, target_size=(256, 192,3))
  
  # convert the image pixels to a numpy array
  image = img_to_array(image)

  # reshape data for the model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

  # prepare the image for the VGG model
  image = preprocess_input(image)

  # load weights into new model
  model = define_model()

  #cargamos el modelo guardado
  model.load_weights(modelo)

  #ejecutamos la prediccion
  pred = model.predict(image, batch_size=1, verbose=True)
  pred[pred > 0.5] = 1.
  pred[pred <= 0.5] = 0.

  if pred[0][0]==1:
    return True
  else:
    return False

'''
Generar mascaras a partir de 2 imagenes MRI, T1 y FLAIR, usando un modelo precargao y dano una ruta 
de salida donde grabaremos la imagen resultante mascara
Ej: 
T1_image = 'mPPMI_3664_MR_SAG_T1_3DMPRAGE__br_raw_20130321141121355_9_S184907_I363981.nii' #'./input/pre/FLAIR.nii.gz'#sys.argv[1] #absolute path of the flair image.
FLAIR_image = 'mrPPMI_3664_MR_AX_FLAIR_br_raw_20130321141116659_8_S184906_I363980.nii'#'./input/pre/T1.nii.gz'#sys.argv[2] #absolute path of the t1 image.
    
model_path = './sample_data/Parkison-s-disease-/models/unet2/unet2.h5'
MASK_image =  'M_3664.nii.gz' 
MASK_folder =  '/content/media/MascaraBinaria' 
generateMaskTwoArgument(FLAIR_image, T1_image, model_path, MASK_image,MASK_folder)
'''
def generateMaskTwoArgument(FLAIR_image_path, T1_image_path, model_path, output_path,pathFolderfinal):

    FLAIR_image = sitk.ReadImage(FLAIR_image_path)
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    
    T1_image = sitk.ReadImage(T1_image_path)
    T1_array = sitk.GetArrayFromImage(FLAIR_image)
    
    img_two_channels = general_preprocessing(FLAIR_array, T1_array)
    img_shape = (rows_standard, cols_standard, 2)
    model = get_unet_2(img_shape)
    #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
    model.load_weights(model_path)
    pred = model.predict(img_two_channels, batch_size=1, verbose=True)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.
    
    original_pred = general_postprocessing(FLAIR_array, pred)
    mask_new = sitk.GetImageFromArray(original_pred)
    mask_new.CopyInformation(FLAIR_image)
    #print("Inicio------------------------------------------------------------->")
    for k in FLAIR_image.GetMetaDataKeys():
      v = FLAIR_image.GetMetaData(k)
      mask_new.SetMetaData(k,v)
      #print("({0}) = = \"{1}\"".format(k,v))
    #print("Fin------------------------------------------------------------->")
    print("Inicio------------------------------------------------------------->")
    for k in mask_new.GetMetaDataKeys():
      v = mask_new.GetMetaData(k)
      print("({0}) = = \"{1}\"".format(k,v))
    print("Fin------------------------------------------------------------->")
    final_output_path=pathFolderfinal+output_path
    sitk.WriteImage(mask_new, final_output_path )

'''
Generar mascara usando como referencia image MRI Flair y mascara binaria
Ej:getMascaraIntensidad('/content/mrESANDI_ITOIZ_JUAN_JOSE_t2_tirm_TRA_dark-fluid_3mm_20110524164030_9.nii.gz','/content/mjuanjose.nii','/content/media/MascarasIntensidades')
'''
def getMascaraIntensidad(Flair,MaskB,pathFolderfinal):
  FLAIR_image_path = Flair  
  MASK_image_path =  MaskB

  FLAIR_image = sitk.ReadImage(FLAIR_image_path,sitk.sitkInt32)
  MASK_image = sitk.ReadImage(MASK_image_path,sitk.sitkInt32)
  FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
  MASK_array = sitk.GetArrayFromImage(MASK_image)

  dotProduct = np.multiply(FLAIR_array,MASK_array)
  mask_new = sitk.GetImageFromArray(dotProduct)
  mask_new.CopyInformation(FLAIR_image)
  #print("Inicio------------------------------------------------------------->")
  for k in FLAIR_image.GetMetaDataKeys():
    v = FLAIR_image.GetMetaData(k)
    mask_new.SetMetaData(k,v)
    #print("({0}) = = \"{1}\"".format(k,v))
  List_MASK_image_path=MASK_image_path.split("/")
  #print(List_MASK_image_path[2])
  filename_resultImage = pathFolderfinal+"Product_" + List_MASK_image_path[2]
  #sitk.WriteImage(mask_new, "media/ImagenesMascaras/ProductP_3415.nii.gz" )
  sitk.WriteImage(mask_new, filename_resultImage)


'''
Obtener el overlay entre la mascara de intensidades y la MRI FLAIR

'''
def getOverlay(FLAIR_image_path,MASK_image_path,PathNameImageOverlay):

    color = [252,132,13]

    #FLAIR_image_path = './input/pre/mrESANDI_ITOIZ_JUAN_JOSE_t2_tirm_TRA_dark-fluid_3mm_20110524164030_9.nii.gz'#'./input/pre/T1.nii.gz'#sys.argv[2] #absolute path of the t1 image.    
    #MASK_image_path =  '/content/Parkison-s-disease-/output/HM_EsantiJuanJose.nii.gz'

    FLAIR_image = sitk.ReadImage(FLAIR_image_path,sitk.sitkInt32)
    MASK_image = sitk.ReadImage(MASK_image_path,sitk.sitkInt32)
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    MASK_array = sitk.GetArrayFromImage(MASK_image)

    #PROCESO DE OVERLAY DE LAS IMAGENES
    resampled_FLAIR = sitk.Resample(FLAIR_image, MASK_image, sitk.Transform(),sitk.sitkLinear, 0.0, FLAIR_image.GetPixelID())
    rescaled_FLAIR = sitk.Cast(sitk.RescaleIntensity(resampled_FLAIR), sitk.sitkUInt8)

    #GUARDO EL OVERLAY ENTRE LAS IMAGENES
    sitk.WriteImage(sitk.LabelOverlay(rescaled_FLAIR, MASK_image, 0.5,colormap=color), PathNameImageOverlay) 


'''
Funcion que convierte un folder que contiene imagenes dicom en un solo archivo nii
Ej:dcm2nii("/content/dcm","/content/nii")
por lo general se crea un nombre automatico del archivo
'''
def dcm2nii(Pathdcmfiles,PathoutputFolder):
  try:
    dicom2nifti.convert_directory(Pathdcmfiles,PathoutputFolder, compression=True, reorient=True)
    return 1
  except:
    #print("An exception occurred")
    return -1
#**************************************************MODELOS DE CLASIFICACION ML*****************************************
# cargar la base
def load_dataset(full_path):
    # cargar como numpy array
    data = read_csv(full_path, header=None)
    data = data.values
    # split input - output 
    X, y = data[2:, :-1], data[2:, -1]
    # clases 0 y 1
    y = LabelEncoder().fit_transform(y)
    return X, y
# evaluar el modelo
def evaluate_model(X, y, model):
    # procedimiento de evaluación
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluar modelo
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    
    return scores

# definir modelos a probar
def get_models():
    models, names = list(), list()
    # Regresión Logística
    models.append(LogisticRegression(solver='lbfgs', class_weight='balanced'))
    names.append('LR')
    # SVM
    models.append(SVC(gamma="scale", class_weight='balanced'))
    names.append('SVM')
    # Random Forest
    models.append(RandomForestClassifier(n_estimators=1000))
    names.append('RF')
    #Naive Bayes
    models.append(GaussianNB())
    names.append("NB")
    #Arbol de decisiones
    models.append(DecisionTreeClassifier(criterion='entropy'))
    names.append("DT")
    #K vecinos
    models.append(KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))
    names.append("KN")
    return models, names
#Evaluando modelo en entrenamiento
def medirPresicionScoresModelos(X,y):
    # definir modelos
    models, names = get_models()
    results = list()
    namesEnd = list()
    resultsEnd = list()
    resultsEnd1 = list()
    # evaluar cada modelo
    for i in range(len(models)):
        scores = evaluate_model(X, y, models[i])
        results.append(scores)
        namesEnd.append(names[i])
        resultsEnd.append(mean(scores))
        resultsEnd1.append(std(scores))
        # resumen
        #print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
    return namesEnd,resultsEnd,resultsEnd1
#Separando el target (y) y data(X) del archivo CSV
def SeparaDataCsv(pathFull): 
    X,y=load_dataset(pathFull) 
    #NORMALIZO LA DATA
    X = normalize(X, axis=0)
    return X,y
#Guardando el mejor modelo de ML para clasificacion de PD
def guardarMejorModeloML(fileNameModeloSavePKL):
      # definir modelos
      models, names = get_models()
      resultsEnd = list()
      # evaluar cada modelo
      for i in range(len(models)):
          scores = evaluate_model(X, y, models[i])
          resultsEnd.append(mean(scores))
      #Separo los datos de "train" en entrenamiento y prueba para probar el algoritmo de Regresion Logistica
      X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
      #Se escalan todos los datos
      escalar = StandardScaler()
      X_train = escalar.fit_transform(X_train)
      X_test = escalar.transform(X_test)
      y_pred = 0.0
      algoritmo=""
      for i in range(len(resultsEnd)):
        if(resultsEnd[i]==max(resultsEnd)):
          algoritmo=models[i]
          print(models[i])
          #Entreno el modelo
          algoritmo.fit(X_train,y_train)
          #Realizo una prediccion
          y_pred=algoritmo.predict(X_test)
          print('test:',y_test)
          print('Prediccion:',y_pred)
          #joblib.dump(algoritmo,fileNameModeloSavePKL)
#Diagnostico PD mediante datos de caracteristicas de archivo CSV
def diagnosticoFinalPD_Habla(StrFeaturesToPredict):


  listaDataFeatures=StrFeaturesToPredict.split('","') #lista de string separado por ","
  print(len(listaDataFeatures))
  print(listaDataFeatures)
  if(len(listaDataFeatures) <=1):
    return  "-- %" , "-- %"
  strLista=" ".join(listaDataFeatures)#Unimos los lementos de la lista por vacio
  strLista=strLista.replace('"',' ') #reemplazamos la comilla " por espacio vacio
  listaDataFeatures=strLista.split(' ') #obtenemos una nueva lista separada por vacios
  listaDataFeatures.pop(0) #se elimina el primer elemento de la lista porque es un espacio vacio
  listaDataFeatures.pop(-1) #se elimina el ultimo elemento de la lista porque es un espacio vacio
  listaDataFeatures2=[listaDataFeatures]#Lista de lista de datos de las features
  #print(listaDataFeatures2)
  clf_from_joblib=joblib.load('media/models/ClasificadorPDHabla_202.pkl')
  pred2=clf_from_joblib.predict(listaDataFeatures2)
  #print(pred2)
  #print(clf_from_joblib.predict_proba(listaDataFeatures2))
  porcentajeH=clf_from_joblib.predict_proba(listaDataFeatures2)[0,0]
  #print(porcentajeH)
  porcentajePD=clf_from_joblib.predict_proba(listaDataFeatures2)[0,1]
  #print(porcentajePD)

  return str(round(porcentajePD, 2)*100) + "%" ,str(round(porcentajeH,2)*100) + "%"
def diagnosticoFinalPD_MRI(listFeaturesToPredict):
  listFeaturesToPredict1=list()
  for i in listFeaturesToPredict:
    lista1=list(i)
    #print(list())
  lista1.pop(0)
  listFeaturesToPredict1.append(lista1) #convierto la tupla de features a lista de lista de features
  #print(listFeaturesToPredict1)
  clf_from_joblib=joblib.load('media/models/ClasificadorPDMRI_49.pkl')
  pred2=clf_from_joblib.predict(listFeaturesToPredict1)
  print(pred2)
  #print(clf_from_joblib.predict_proba(listFeaturesToPredict1))
  porcentajeH=clf_from_joblib.predict_proba(listFeaturesToPredict1)[0,0]
  print(porcentajeH)
  porcentajePD=clf_from_joblib.predict_proba(listFeaturesToPredict1)[0,1]
  print(porcentajePD)

  return str(round(porcentajePD, 2)*100) + "%" ,str(round(porcentajeH,2)*100) + "%"
#View para calcular las precisiones de los modelos de ML para la clasificacion
def precisionesCsv_Habla(request):
    #print("ONSERVA BRO",CSV.objects.last())
    #names=''
    #results = list()
    #results1 = list()
    if(CSV.objects.last()):
        instanciaCsv=CSV.objects.last()
        #print("entro",CSV.objects.last())
        path="media/"+str(instanciaCsv.documento)
        X,y=SeparaDataCsv(path)
        names,results,results1=medirPresicionScoresModelos(X,y)
        if request.method == 'GET':       
            return render(request, 'Diagnostico/resultML_habla.html',{'data':zip(names,results, results1)}) 
        messages.success(request, 'El proceso se ha completado con éxito.')
        return redirect('home_administrador')
    else:
        if request.method == 'GET':       
            return render(request, 'Diagnostico/resultML_habla.html') 
        messages.error(request, 'El proceso no se ha completado con éxito.')
        return redirect('home_administrador')
#View para ingresar data para diagnosticar PD con el modelo de clasifcicacion de ML
def pruebasMLCsv_Habla(request):
    if request.method == 'GET':       
        return render(request, 'Diagnostico/IngresoDataToValid.html') 
    return redirect('home_administrador')
#View para calcular el diagnostico de PD
def diagnoticoPorCsv_Habla(request):
    #path="media/models/ClasificadorPDHabla.pkl"
    #NameModel="ClasificadorPDHabla.pkl"
    global q
    q = request.GET.get('q','')
    #print('INICIO',q,'FIN')
    diagnostico1,diagnostico2=diagnosticoFinalPD_Habla(q)
    #diagnosticoFinalPD_Habla(q)
    if(diagnostico1=="-- %" and diagnostico2=="-- %"):
        messages.error(request, 'No se puede llevar a cabo el diagnostico, ingreso dato valido!')
        return redirect('home_administrador')
    if request.method == 'GET': 
        print(diagnostico1)
        print(diagnostico2)
        messages.success(request, 'Su diagnóstico se ha completado con éxito.')       
        return render(request, 'Diagnostico/diagnosticoHabla.html',{'data1':diagnostico1,'data2':diagnostico2}) 

    return redirect('home_administrador')

#View para barra de carga, simulando todo el tiempo que tarda en dar el modelo un diagnostico
def barraCargaModeloDiagPorMRI(request):
    if request.method == 'GET':    
        return render(request, 'Diagnostico/barraCargaDiagPD.html') 
    if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
        return redirect('home_doctor')
    return redirect('home_administrador')
#View para calcular el diagnostico de PD por MRI
def diagnoticoPorMRI(request):
    colsNamesFeatures=[
        "1. Nombre",
        "2. Curtosis (intensidad)",
        "3. Redondez (intensidad)",
        "4. Perimetro[nm] (intensidad)",
        "5. Varianza[nm^2] (intensidad)",
        "6. Desviación estandar[nm] (intensidad)",
        "7. Volumen[nm^3] (forma)",
        "8. Elongación[nm] (forma)",
        "9. Flatness (forma)",
        "10. Cantidad lesiones"]

    diagnosticoPD = 0
    diagnosticoSano = 0
    print("Existe una T1 en la BD? ",ImagenMRI.objects.filter(tipo='T1').exists())
    if(ImagenMRI.objects.filter(tipo='T1').exists()==False or ImagenMRI.objects.filter(tipo='FLAIR').exists()==False):

        if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
            messages.error(request, 'Su diagnóstico, no se a podido completar..Intentalo de nuevo!')
            return redirect('home_doctor')
        messages.error(request, 'Su diagnóstico, no se a podido completar,Asegurese de ingresar las imagenes MRI primero...Intentalo de nuevo!')
        return redirect('home_administrador')

    instanciaUltimaT1=ImagenMRI.objects.filter(tipo='T1').last()
    instanciaUltimaFLAIR=ImagenMRI.objects.filter(tipo='FLAIR').last()
    instanciaSujeto=Sujeto.objects.get(id_sujeto=instanciaUltimaT1.id_sujeto_id)

    T1_image = instanciaUltimaT1.imagen.path
    FLAIR_image = instanciaUltimaFLAIR.imagen.path

    MASK_image_Name = 'M_'+ str(instanciaSujeto.nombre) + str(instanciaSujeto.apellido) + '.nii'  #debe ser en .nii para que funcione rapido y bien
    generateMaskTwoArgument(FLAIR_image, T1_image, model_path, MASK_image_Name,Path_MASK_folder1) #Genera mascara binaria
    print("VERIFICA=============>",request.method)
    if request.method == 'GET': 
        #generateMaskTwoArgument(FLAIR_image, T1_image, model_path, MASK_image_Name,Path_MASK_folder1) #Genera mascara binaria
        Path_MASK_folder = 'media/ImagenesMascarasBinarias/'+MASK_image_Name
        getMascaraIntensidad(FLAIR_image,Path_MASK_folder,Path_MASK_folder_Int) #Genera mascara con intensidades corregida

        Final_MASK_image_path='media/ImagenesMascaras/Product_'+MASK_image_Name
        name=str(instanciaSujeto.nombre) +' '+ str(instanciaSujeto.apellido)

        features=extraerCaracteristicas(Final_MASK_image_path,name) #Extraccion de caracteristicas
        for i in features:
            listFeaturesToPredict=list(i) #Lista de los valores de las caracteristicas calculados
        print(listFeaturesToPredict)   
        diagnosticoPD,diagnosticoSano=diagnosticoFinalPD_MRI(features) #Calcula el diagnostico de PD o NO PD
        PathNameImageOverlay='media/ImagenesOverlay/Overlay_'+MASK_image_Name
        getOverlay(FLAIR_image,Final_MASK_image_path,PathNameImageOverlay)
        print(diagnosticoPD)
        print(diagnosticoSano)
        Final_MASK_image_path_SaveTable='/ImagenesMascaras/Product_'+MASK_image_Name
        PathNameImageOverlay_SaveTable='/ImagenesOverlay/Overlay_'+MASK_image_Name
        #Guardamos en las respectivas tablas de la BD la infor de todo el proceso realizado
        print(ImagenMascara.objects.filter(imagen=Final_MASK_image_path_SaveTable).exists())
        print(Overlay.objects.filter(imagen=PathNameImageOverlay_SaveTable).exists())
        
        if(ImagenMascara.objects.filter(imagen=Final_MASK_image_path_SaveTable).exists()==False):
            ImagenMascara.objects.create(imagen=Final_MASK_image_path_SaveTable)

        instanciaMask=ImagenMascara.objects.filter(imagen=Final_MASK_image_path_SaveTable).last()

        if(Overlay.objects.filter(imagen=PathNameImageOverlay_SaveTable).exists()==False):
            Overlay.objects.create(imagen=PathNameImageOverlay_SaveTable,id_mri=instanciaUltimaFLAIR,id_mask=instanciaMask)

        '''print(TablaCaracteristicas.objects.filter(curtosisI=listFeaturesToPredict[1]).exists(),
            TablaCaracteristicas.objects.filter(redondezI=listFeaturesToPredict[2]).exists(),
            TablaCaracteristicas.objects.filter(perimetroI=listFeaturesToPredict[3]).exists(),
            TablaCaracteristicas.objects.filter(varianzaI=listFeaturesToPredict[4]).exists(),
            TablaCaracteristicas.objects.filter(desviEstandI=listFeaturesToPredict[5]).exists(),
            TablaCaracteristicas.objects.filter(volumenF=listFeaturesToPredict[6]).exists(),
            TablaCaracteristicas.objects.filter(enlogacionF=listFeaturesToPredict[7]).exists(),
            TablaCaracteristicas.objects.filter(flagnessF=listFeaturesToPredict[8]).exists())'''

        if((TablaCaracteristicas.objects.filter(curtosisI=listFeaturesToPredict[1]).exists()==False) and
            (TablaCaracteristicas.objects.filter(redondezI=listFeaturesToPredict[2]).exists()==False) and
            (TablaCaracteristicas.objects.filter(perimetroI=listFeaturesToPredict[3]).exists()==False) and
            (TablaCaracteristicas.objects.filter(varianzaI=listFeaturesToPredict[4]).exists()==False) and
            (TablaCaracteristicas.objects.filter(desviEstandI=listFeaturesToPredict[5]).exists()==False) and
            (TablaCaracteristicas.objects.filter(volumenF=listFeaturesToPredict[6]).exists()==False) and
            (TablaCaracteristicas.objects.filter(enlogacionF=listFeaturesToPredict[7]).exists()==False) and
            (TablaCaracteristicas.objects.filter(flagnessF=listFeaturesToPredict[8]).exists()==False)):

            TablaCaracteristicas.objects.create(nombrePaciente=listFeaturesToPredict[0],curtosisI=listFeaturesToPredict[1],redondezI=listFeaturesToPredict[2],
            perimetroI=listFeaturesToPredict[3],varianzaI=listFeaturesToPredict[4],desviEstandI=listFeaturesToPredict[5],volumenF=listFeaturesToPredict[6],enlogacionF=listFeaturesToPredict[7],flagnessF=listFeaturesToPredict[8],cantidad=listFeaturesToPredict[9],
            id_sujeto=instanciaSujeto,id_mask=instanciaMask)

        instanciaOverlay=Overlay.objects.filter(imagen=PathNameImageOverlay_SaveTable).last()
        instanciaTablaC=TablaCaracteristicas.objects.last()
        if((Diagnostico.objects.filter(id_mri=instanciaUltimaFLAIR).exists()==False) and
        (Diagnostico.objects.filter(id_mask=instanciaMask).exists()==False) and
        (Diagnostico.objects.filter(id_overlay=instanciaOverlay).exists()==False) and
        (Diagnostico.objects.filter(id_sujeto=instanciaSujeto).exists()==False) and
        (Diagnostico.objects.filter(id_tablaC=instanciaTablaC).exists()==False)):

            Diagnostico.objects.create(id_mri=instanciaUltimaFLAIR,id_mask=instanciaMask,id_overlay=instanciaOverlay,id_sujeto=instanciaSujeto,id_tablaC=instanciaTablaC,porcentPD=diagnosticoPD,porcentNoPD=diagnosticoSano)
        instanciaDiag=Diagnostico.objects.filter(id_sujeto=instanciaSujeto)[0]
        #print(listFeaturesToPredict[0],listFeaturesToPredict[1],listFeaturesToPredict[2],listFeaturesToPredict[3],listFeaturesToPredict[4],listFeaturesToPredict[5],listFeaturesToPredict[6],listFeaturesToPredict[7],listFeaturesToPredict[8],listFeaturesToPredict[9])
        if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
            messages.success(request, 'Su diagnóstico se ha completado con éxito.')
            return render(request, 'Diagnostico/diagnosticoPD_MRI_toDoctor.html',{'data1PD':diagnosticoPD,'data2Sano':diagnosticoSano,'Overlay_image':PathNameImageOverlay,'listFeaturesToPredict':zip(listFeaturesToPredict,colsNamesFeatures),'diagnostico':instanciaDiag}) 
        messages.success(request, 'Su diagnóstico se ha completado con éxito.')
        return render(request, 'Diagnostico/diagnosticoPD_MRI.html',{'data1PD':diagnosticoPD,'data2Sano':diagnosticoSano,'Overlay_image':PathNameImageOverlay,'listFeaturesToPredict':zip(listFeaturesToPredict,colsNamesFeatures),'diagnostico':instanciaDiag}) 
    print(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists())
    if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
        #messages.success(request, 'Su diagnóstico se ha completado con éxito.')
        return redirect('home_doctor')
    #messages.success(request, 'Su diagnóstico se ha completado con éxito.')
    return redirect('home_administrador')

#View para mostrar lista de diagnosticos realizados
class DiagnosticoDisponible(ListView):
    model = Diagnostico
    template_name = 'Diagnostico/listaDiagnosticoDispo.html'
    success_url = reverse_lazy('diagnostico_disponible')
    paginate_by=5
class DiagnosticoDisponibleToDoctor(ListView):
    model = Diagnostico
    template_name = 'Diagnostico/listaDiagnosticoDispo_toDoctor.html'
    success_url = reverse_lazy('diagnostico_disponibleToDoctor')
    paginate_by=5
#View para eliminar diagnostico de un paciente 
def RegistroDiagDelete(request, id_diag):
    instancia = Diagnostico.objects.get(id_diag=id_diag)
    if request.method == 'POST':        
        instancia.delete()
        if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
            messages.success(request, 'Su registro de diagnóstico ha sido eliminado con éxito.')
            return redirect('diagnostico_disponibleToDoctor')
        messages.success(request, 'Su registro de diagnóstico ha sido eliminado con éxito.')
        return redirect('diagnostico_disponible')
    if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
        return render(request, 'Diagnostico/eliminarRegistroDiagToDoctor.html',{'instancia':instancia})
    return render(request, 'Diagnostico/eliminarRegistroDiag.html',{'instancia':instancia})
#View para eliminar diagnostico de un paciente 
def EditarDiagObser(request, id_diag):
    form1 = DiagnosticoForm(request.POST)#form1---->descripcion y es_parkinson
    instancia = Diagnostico.objects.get(id_diag=id_diag)
    if request.method == 'POST':
        instancia.descripcion = request.POST.get('descripcion', None)
        opcion=request.POST.get('is_parkinson', None)
        print("OBSERVA: ",instancia,opcion)
        if(opcion=="Si"):
            instancia.is_parkinson = True
        else:
            instancia.is_parkinson = False
        #print(instancia.descripcion,instancia.is_parkinson)
        #print("\nAqui esta la descripcion y si es parkinson")
        #print(request.POST.get('descripcion', None),request.POST.get('is_parkinson', None))
        instancia.save()
        if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
            messages.success(request, 'Sus observaciones del diagnóstico han sido registradas con éxito.')
            #return redirect('diagnoticoPorMRI')
            return redirect('diagnostico_disponibleToDoctor')
        messages.success(request, 'Sus observaciones del diagnóstico han sido registradas con éxito.')
        #return redirect('diagnoticoPorMRI')
        return redirect('diagnostico_disponible')
    if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
        return render(request, 'Diagnostico/diagnosticoPD_MRI_toDoctor.html',{'instancia':instancia})
    return render(request, 'Diagnostico/diagnosticoPD_MRI.html',{'instancia':instancia})

class EditarDiagnostico(DeleteView):
    model = Diagnostico
    template_name = 'Diagnostico/editarRegistroDiag.html'
    success_url = reverse_lazy('diagnostico_disponible')
class EditarDiagnosticoToDoctor(DeleteView):
    model = Diagnostico
    template_name = 'Diagnostico/editarRegistroDiagToDoctor.html'
    success_url = reverse_lazy('diagnostico_disponible')
class EliminarDiagnostico(DeleteView):
    model = Diagnostico
    template_name = 'Diagnostico/eliminarRegistroDiag.html'
    success_url = reverse_lazy('diagnostico_disponible')
class EliminarDiagnosticoToDoctor(DeleteView):
    model = Diagnostico
    template_name = 'Diagnostico/eliminarRegistroDiagToDoctor.html'
    success_url = reverse_lazy('diagnostico_disponibleToDoctor')
#View para buscar un diagnostico nombre de paciente
def busquedaDiagByPaciente(request):
    q = request.GET.get('q','')
    pacienteID=0
    if(Sujeto.objects.filter(nombre__startswith=q)):
        pacienteID=Sujeto.objects.get(nombre__startswith=q).id_sujeto
    if(Sujeto.objects.filter(apellido__startswith=q)):
        pacienteID=Sujeto.objects.get(apellido__startswith=q).id_sujeto

    DiagBypacienteID=Diagnostico.objects.filter(id_sujeto=pacienteID)
    if(DiagBypacienteID and q!=""):
      diagnosticos=DiagBypacienteID
    else:
      diagnosticos = ""

    if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
        return render(request, 'Diagnostico/listaDiagnosticoDispo_busqueda_toDoctor.html', {'diagnosticos': diagnosticos ,'busqueda':q})
    return render(request, 'Diagnostico/listaDiagnosticoDispo_busqueda.html', {'diagnosticos': diagnosticos ,'busqueda':q})

 #view feeback de usuario opcion cargar MRI
def feedback_CargarMRI(request):
    return render(request,'Diagnostico/feedback_CargarMRI.html')

 #view feeback de usuario opcion cargar CSV
def feedback_CargarCSV(request):
    return render(request,'Diagnostico/feedback_CargarCSV.html')

 #view feeback de usuario opcion diagnostico por habla
def feedback_DiagHabla(request):
    return render(request,'Diagnostico/feedback_DiagHabla.html')

 #view feeback de usuario opcion diagnostico por MRI
def feedback_DiagMri(request):
    return render(request,'Diagnostico/feedback_DiagMRI.html')

 #view feeback de usuario opcion diagnostico por MRI
def feedback_ListDiagMri(request):
    return render(request,'Diagnostico/feedback_ListDiag.html')
def feedback_ListDiagMriToDoctor(request):
    return render(request,'Diagnostico/feedback_ListDiagToDoctor.html')

 #view feeback de usuario opcion precisiones de los modelos ML para estudio del Habla
def feedback_PrecisionesModelsML(request):
    return render(request,'Diagnostico/barraCargaPrecisionesML.html')