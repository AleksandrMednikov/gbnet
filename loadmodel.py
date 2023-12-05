from keras.models import load_model
import numpy as np
import cv2


def model_lun_brain_medniq(filepath, modelpath):
    values = ['brain', 'lungs']
    image = cv2.imread(filepath)
    image = cv2.resize(image, (128, 128))
    image = image/255
    loaded_model = load_model(modelpath)
    predictions = loaded_model.predict([image])
    
    #порог
    if predictions[1]>0.6455790400505066:
        indx = 1
    else:
        indx = 0

    return values[indx]

