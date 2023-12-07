import numpy as np
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Adicione esta linha

# Resto do seu código...


# Diretório contendo suas imagens
data_dir = '/home/iza/Área de Trabalho/n2_ica/imag'

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=False
)

class_counts = np.sum(train_generator.labels, axis=0, dtype=np.float32)
total_samples = np.sum(class_counts)

class_weights = {i: total_samples / (2.0 * class_counts[i]) for i in range(len(class_counts))}

# Treino (80%) e validação (20%)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    class_weights=class_weights,
    shuffle=True 
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical', 
    subset='validation'
)

# CNN pré-treinada VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# adaptação do modelo para duas classes
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# congelamento das camadas pré-treinadas
for layer in base_model.layers:
    layer.trainable = False

# compilação do modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# treinamento do modelo
model.fit(train_generator, epochs=5, validation_data=validation_generator)

# adição de camadas de atenção
attention_layer = Attention()([base_model.output, model.layers[-2].output])
output_with_attention = tf.keras.layers.multiply([base_model.output, attention_layer])

# criando um novo modelo com camadas de atenção
model_with_attention = Model(inputs=base_model.input, outputs=output_with_attention)

# visualize o modelo com camadas de atenção
model_with_attention.summary()
