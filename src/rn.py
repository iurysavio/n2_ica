import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    return features.flatten()

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

data_dir = '/home/iza/Área de Trabalho/n2_ica/imag'

features_list = []
labels_list = []

classes = ['AVCH', 'normal']
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        
        # extrair características e adicionar à lista
        features = extract_features(img_path, base_model)
        features_list.append(features)
        labels_list.append(class_name)

features_array = np.array(features_list)
labels_array = np.array(labels_list)

df = pd.DataFrame(features_array)
df['label'] = labels_array

df.to_csv('features_and_labels.csv', index=False)
