from django.shortcuts import render
import os
import sys
import time
import numpy as np
import dicom2nifti
import dicom2nifti.settings as settings
import warnings
import scipy
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation, Dense, Flatten
from keras.activations import elu
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
from PIL import Image
from resizeimage import resizeimage
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential, Model
import numpy as np
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

# Create your views here.

def mask_X_size(file):
    image_path = sitk.ReadImage(file)
    image_path_array = sitk.GetArrayFromImage(image_path)
    return image_path_array.shape[0]

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
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    print("aqui muestra canales o capas")
    print(cw)
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
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
        original_pred[:,(image_rows_Dataset-rows_standard)/2:(image_rows_Dataset+rows_standard)/2,(image_cols_Dataset-cols_standard)/2:(image_cols_Dataset+cols_standard)/2] = pred[:,:,:,0]
        
        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred
    elif (image_rows_Dataset >= rows_standard and image_cols_Dataset < cols_standard):
        original_pred[:, (image_rows_Dataset-rows_standard)/2:(image_rows_Dataset+rows_standard)/2,:] = pred[:,:, (cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2,0]

        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred


    elif (image_rows_Dataset < rows_standard and image_cols_Dataset >= cols_standard):
        original_pred[:, :,(image_cols_Dataset-cols_standard)/2:(image_cols_Dataset+cols_standard)/2] = pred[:,(rows_standard-image_rows_Dataset)/2:(rows_standard+image_rows_Dataset)/2,:,0]

        original_pred[0:start_slice, :, :] = 0
        original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
        return original_pred

    else:
        original_pred = pred[:,(rows_standard-image_rows_Dataset)/2:(rows_standard+image_rows_Dataset)/2,(cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2,0]

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
        FLAIR_image = FLAIR_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
        brain_mask_FLAIR = brain_mask_FLAIR[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
        ###------Gaussion Normalization here
        np.subtract(FLAIR_image, np.mean(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
        np.divide(FLAIR_image, np.std(FLAIR_image[brain_mask_FLAIR == 1]), out=FLAIR_image, casting="unsafe")#FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        T1_image = T1_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
        brain_mask_T1 = brain_mask_T1[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
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
        FLAIR_image_suitable[:, :, (cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = FLAIR_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), :]
    
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
    
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        np.subtract(T1_image, np.mean(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
        np.divide(T1_image, np.std(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:, :, (cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = T1_image[:, (image_rows_Dataset/2-rows_standard/2):(image_rows_Dataset/2+rows_standard/2), :]
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
        FLAIR_image_suitable[:, (rows_standard - image_rows_Dataset)/2:(rows_standard + image_rows_Dataset)/2,:] = FLAIR_image[:, :, (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
    
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
    
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        np.subtract(T1_image, np.mean(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
        np.divide(T1_image, np.std(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:,(rows_standard - image_rows_Dataset)/2:(rows_standard + image_rows_Dataset)/2,:] = T1_image[:, :, (image_cols_Dataset/2-cols_standard/2):(image_cols_Dataset/2+cols_standard/2)]
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
        FLAIR_image_suitable[:, (rows_standard - image_rows_Dataset)/2:(rows_standard + image_rows_Dataset)/2,(cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = FLAIR_image[...]
    
        # T1 -----------------------------------------------
        brain_mask_T1[T1_image >=thresh_T1] = 1
        brain_mask_T1[T1_image < thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
    
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
            #------Gaussion Normalization
        np.subtract(T1_image, np.mean(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
        np.divide(T1_image, np.std(T1_image[brain_mask_T1 == 1]), out=T1_image, casting="unsafe")#T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:, (rows_standard - image_rows_Dataset)/2:(rows_standard + image_rows_Dataset)/2,(cols_standard-image_cols_Dataset)/2:(cols_standard+image_cols_Dataset)/2] = T1_image[...]
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

  stats_list = [(intensity_stats.GetKurtosis(i),
                intensity_stats.GetPhysicalSize(i),
                intensity_stats.GetRoundness(i),
                intensity_stats.GetPerimeter(i),
                intensity_stats.GetVariance(i),
                #intensity_stats.GetElongation(i),
                intensity_stats.GetStandardDeviation(i),
                  shape_stats.GetPhysicalSize(i),
                  shape_stats.GetElongation(i),
                  shape_stats.GetFlatness(i)) for i in intensity_stats.GetLabels()]
  cols=["Curtosis_intensidad",
        "Área_intensidad",
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
  print(xpru)
  stats_list = [ (
                              nombrePaciente,
                              xpru[0],
                              xpru[1],
                              xpru[2],
                              xpru[3],
                              xpru[4],
                              xpru[5],
                              xpru[6],
                              xpru[7],
                              xpru[8],
                              intensity_stats.GetNumberOfLabels())]
  return stats_list

'''
Guarda una imagen nifty en jpg; si tiene un solo corte en z se lo toma caso contrario
se calcula la mitad del corte en z y luego es procesado como un solo slice
Ej: saveMRINiftytoJPG("./mr1.3.12.2.1107.5.2.32.35170.201102231753385275793382.0.0.0_t2_tirm_TRA_dark-fluid_3mm_20110223165745_9.nii.gz",".","mr1.3.12.2.1107.5.2.32.35170.201102231753385275793382.0.0.0_t2_tirm_TRA_dark-fluid_3mm_20110223165745_9")
'''
def saveMRINiftytoJPG(PathSource,PathFolderfinal,Finalname):
  grid_image = sitk.ReadImage(PathSource)
  nda = sitk.GetArrayFromImage(grid_image)
  z=nda.shape[0]
  if z==1:
    os.system("med2image -i "+str(PathSource)+" -d "+str(PathFolderfinal)+" -o "+str(Finalname)+ " --outputFileType jpg")
    #med2image -i PathSource -d PathFolderfinal -o Finalname --outputFileType jpg 
    return 1
  elif z>1:
    n=int(z/2)
    os.system("med2image -i "+str(PathSource)+" -d "+str(PathFolderfinal)+" -o "+ str(Finalname)+ " --outputFileType jpg --sliceToConvert "+str(n))
    #med2image -i PathSource -d PathFolderfinal -o Finalname --outputFileType jpg --sliceToConvert n
    return 1
  else:
    return -1


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
def getMascaraIntensidad(Flair,MaskB,pathFolderfinal)
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
  filename_resultImage = pathFolderfinal+"ProductP" + MASK_image_path
  sitk.WriteImage(mask_new, filename_resultImage )

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
