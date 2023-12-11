import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, ResNet50, VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    return features.flatten()

models = {
    'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(512, 512, 3)),
    'InceptionResNetV2': InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(512, 512, 3)),
    'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3)),
    'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3)),
}

data_dir = '/home/iza/Área de Trabalho/DisciplicaICA/n2_ica/imag' 

for model_name, model in models.items():
    features_list = []
    labels_list = []

    classes = ['AVCH', 'normal']
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            # extrai caracteristicas e adicionar a lista
            features = extract_features(img_path, model)
            features_list.append(features)
            labels_list.append(f'{class_name}_{model_name}')

    features_array = np.array(features_list)
    labels_array = np.array(labels_list)

    df = pd.DataFrame(features_array)
    df['label'] = labels_array

    csv_filename = f'features_and_labels_{model_name}.csv'
    df.to_csv(csv_filename, index=False)
