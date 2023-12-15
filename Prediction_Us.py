#!/usr/bin/env python
# coding: utf-8


# Importation des bibliothèques nécessaires
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout,BatchNormalization,Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Chemin vers les dossiers de données
data_dir = 'C:/Users/bmd tech/Downloads/DATASET_PROJECT'

train_dir = os.path.join(data_dir, 'Train')
test_dir = os.path.join(data_dir, 'Test')
validation_dir = os.path.join(data_dir, 'Validation')


print(os.path.exists(data_dir))  # retourne True si le chemin existe


# consulter les dossier qui sont a l'intérieur du chemin
listi=os.listdir(data_dir)
listi


# ImageDataGenerator pour la normalisation et la division des données
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    class_mode='binary',
)


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    class_mode='binary'
)





class_labels = list(train_generator.class_indices.keys())# pour recuperer les label reels

# Accédez au premier batch d'images
images_batch = train_generator[0][0]
labels_batch = train_generator[0][1]

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images_batch[i])
    plt.title(f"label: {class_labels[int(labels_batch[i])]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

# Définir la fonction de modèle
def build_model(hp):
    model = keras.Sequential()

    # Ajouter une couche d'entrée
    model.add(layers.Flatten(input_shape=(64, 64, 3)))  

    # Boucle pour ajouter des couches cachées avec des hyperparamètres recherchés
    for i in range(hp.Int('num_layers', min_value=1, max_value=5, step=1)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                               activation='relu'))

    # Ajouter la couche de sortie
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compiler le modèle
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model



# Instancier le tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,  # Nombre total d'essais de modèles à essayer
    directory='Tuner_Direct',  # Répertoire où sauvegarder les résultats du tuner
    project_name='Ultrasound_ab')  # Nom du projet pour les résultats du tuner



# Lancer la recherche d'hyperparamètres
tuner.search(train_generator, epochs=5, validation_data=validation_generator)


# Obtenir le meilleur modèle
best_model = tuner.get_best_models(num_models=1)[0]



# Afficher le résumé du modèle
best_model.summary()



# Obtenir les résultats de la recherche
best_trials = tuner.oracle.get_best_trials(num_trials=5)



#Récupérer quelques images et labels de test
test_generator.reset()
batch = next(test_generator)
images, true_labels = batch[0], batch[1]



# Prédictions
predictions_batch = best_model.predict(images)
predicted_labels = np.where(predictions_batch > 0.5, 1, 0)



plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], interpolation='nearest') 
    plt.title(f"True: {class_labels[int(true_labels[i])]}\nPred: {class_labels[int(predicted_labels[i])]}")
    plt.axis('off')
plt.tight_layout() 
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns
# 8. Affichage de la matrice de confusion
# confusion_mtx = confusion_matrix(test_generator.classes, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.title('Matrice de Confusion')
plt.show()


best_model.save('model_US.h5')




