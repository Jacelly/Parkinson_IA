from django.shortcuts import render
import os
from PIL import Image
from resizeimage import resizeimage
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import med2image
import SimpleITK as sitk

# Create your views here.

'''
Guarda una imagen nifty en jpg; si tieneun solocorte en z se lo toma caso contrario
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


# definición del modelo
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
