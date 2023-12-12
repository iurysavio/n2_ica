import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, ResNet50, VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def build_model(model_name, pooltype='max'):
    if model_name == 'VGG16':
        model = VGG16(weights='imagenet', pooling=pooltype, include_top=False)    
    elif model_name == 'InceptionV3':
        model = InceptionV3(weights='imagenet', pooling=pooltype, include_top=False)
    elif model_name == 'InceptionResNetV2':
        model = InceptionResNetV2(weights='imagenet', pooling=pooltype, include_top=False)
    elif model_name == 'ResNet50':
        model = ResNet50(weights='imagenet', pooling=pooltype, include_top=False)
    else:
        raise ValueError(f'Model {model_name} not supported.')

    target_size = 299 if model_name in ['InceptionV3', 'InceptionResNetV2'] else 224
    return model, target_size

def extract_features(img_path, model, target_size):
    img = image.load_img(img_path, target_size=(target_size, target_size))  #Redimensiona a imagem
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    return features.flatten()

models_to_build = ['InceptionV3', 'InceptionResNetV2', 'ResNet50', 'VGG16']

data_dir = '/home/izaquela/Área de Trabalho/n2_ica/src/data'

for model_name in models_to_build:
    model, target_size = build_model(model_name)
    
    features_list = []
    labels_list = []

    classes = ['avch', 'normal']
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            #Extrai características e adiciona à lista
            features = extract_features(img_path, model, target_size)
            labels_list.append(f'{class_name}_{model_name}')
            features_list.append(features)

    features_array = np.array(features_list)
    labels_array = np.array(labels_list)

    df = pd.DataFrame(features_array)
    df['label'] = labels_array

    csv_filename = f'features_and_labels_{model_name}.csv'
    df.to_csv(csv_filename, index=False)
