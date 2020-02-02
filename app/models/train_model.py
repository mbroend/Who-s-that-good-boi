import numpy as np
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.applications.resnet50 import preprocess_input as rn50_preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as irnv2_preproccess_input
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  
from keras.preprocessing import image                  
from tqdm import tqdm
from sklearn.datasets import load_files
from glob import glob
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[35:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
    
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

# Transfer learning using the inceptionresnetv2 trained on imagenet
model_cropped = InceptionResNetV2(weights='imagenet', include_top=False)

# Calculate bottleneck features
bottleneck_features = {}
bottleneck_features['train'] = model_cropped.predict(train_tensors)
bottleneck_features['valid'] = model_cropped.predict(valid_tensors)
bottleneck_features['test'] = model_cropped.predict(test_tensors)

## Creating the ending layers
# Adding dropouts as validation accuracy kept improving, but testing didn't 
doggo_model = Sequential()
doggo_model.add(GlobalAveragePooling2D(input_shape=bottleneck_features['train'].shape[1:]))
doggo_model.add(Dense(133))
doggo_model.add(Dropout(0.4))
doggo_model.add(Dense(133))
doggo_model.add(Dropout(0.4))
doggo_model.add(Dense(133, activation='softmax'))
doggo_model.summary()

# Compiling the model
doggo_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Training the model

epochs = 20
checkpointer = ModelCheckpoint(filepath='weights.best.doggo_model.hdf5', 
                               verbose=1, save_best_only=True)
doggo_model.fit(bottleneck_features['train'], train_targets, 
          validation_data=(bottleneck_features['valid'], valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

# Loading best model
doggo_model.load_weights('weights.best.doggo_model.hdf5')

# Testing 
# get index of predicted dog breed for each image in test set
doggo_predictions = [np.argmax(doggo_model.predict(np.expand_dims(feature, axis=0))) for feature in bottleneck_features['test']]

# report test accuracy
test_accuracy = 100*np.sum(np.array(doggo_predictions)==np.argmax(test_targets, axis=1))/len(doggo_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)