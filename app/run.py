import json
import plotly
import numpy as np
import pandas as pd
import os , io , sys

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Violin, Box
from io import BytesIO
import base64
import tensorflow as tf
import cv2
from keras import backend as k
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.applications.resnet50 import preprocess_input as rn50_preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as irnv2_preproccess_input
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image                  
from tqdm import tqdm
from glob import glob
from PIL import ImageFile, Image                           
ImageFile.LOAD_TRUNCATED_IMAGES = True    
#




app = Flask(__name__,static_url_path='/static')

# Loading dog names in order of doggo_model output
dog_names = ['Affenpinscher', 'Afghan hound', 'Airedale terrier', 'Akita', 'Alaskan malamute', 'American eskimo dog', 'American foxhound', 
             'American staffordshire terrier', 'American water spaniel', 'Anatolian shepherd dog', 'Australian cattle dog', 'Australian shepherd', 
             'Australian terrier', 'Basenji', 'Basset hound', 'Beagle', 'Bearded collie', 'Beauceron', 'Bedlington terrier', 'Belgian malinois', 'Belgian sheepdog', 
             'Belgian tervuren', 'Bernese mountain dog', 'Bichon frise', 'Black and tan coonhound', 'Black russian terrier', 'Bloodhound', 'Bluetick coonhound',
             'Border collie', 'Border terrier', 'Borzoi', 'Boston terrier', 'Bouvier des flandres', 'Boxer', 'Boykin spaniel', 'Briard', 'Brittany', 
             'Brussels griffon', 'Bull terrier', 'Bulldog', 'Bullmastiff', 'Cairn terrier', 'Canaan dog', 'Cane corso','Cardigan welsh corgi', 
             'Cavalier king charles spaniel', 'Chesapeake bay retriever', 'Chihuahua', 'Chinese crested', 'Chinese shar-pei', 'Chow chow', 'Clumber spaniel', 
             'Cocker spaniel', 'Collie', 'Curly-coated retriever', 'Dachshund', 'Dalmatian', 'Dandie dinmont terrier', 'Doberman pinscher', 'Dogue de bordeaux', 
             'English cocker spaniel', 'English setter', 'English springer spaniel', 'English toy spaniel', 'Entlebucher mountain dog', 'Field spaniel', 'Finnish spitz', 
             'Flat-coated retriever', 'French bulldog', 'German pinscher', 'German shepherd dog', 'German shorthaired pointer', 'German wirehaired pointer', 
             'Giant schnauzer', 'Glen of imaal terrier', 'Golden retriever', 'Gordon setter', 'Great dane', 'Great pyrenees', 'Greater swiss mountain dog', 'Greyhound', 
             'Havanese', 'Ibizan hound', 'Icelandic sheepdog', 'Irish red and white setter', 'Irish setter', 'Irish terrier', 'Irish water spaniel', 'Irish wolfhound', 
             'Italian greyhound', 'Japanese chin', 'Keeshond', 'Kerry blue terrier', 'Komondor', 'Kuvasz', 'Labrador retriever', 'Lakeland terrier', 'Leonberger', 
             'Lhasa apso', 'Lowchen', 'Maltese', 'Manchester terrier', 'Mastiff', 'Miniature schnauzer', 'Neapolitan mastiff', 'Newfoundland', 'Norfolk terrier', 
             'Norwegian buhund', 'Norwegian elkhound', 'Norwegian lundehund', 'Norwich terrier', 'Nova scotia duck tolling retriever', 'Old english sheepdog', 
             'Otterhound', 'Papillon', 'Parson russell terrier', 'Pekingese', 'Pembroke welsh corgi', 'Petit basset griffon vendeen', 'Pharaoh hound', 'Plott', 
             'Pointer', 'Pomeranian', 'Poodle', 'Portuguese water dog', 'Saint bernard', 'Silky terrier', 'Smooth fox terrier', 'Tibetan mastiff', 
             'Welsh springer spaniel', 'Wirehaired pointing griffon', 'Xoloitzcuintli', 'Yorkshire terrier']

# Loading models
session = tf.compat.v1.Session(graph=tf.Graph())
with session.graph.as_default():
    k.set_session(session)
    ### Dog/no-dog model ###
    print('Loading resnet50')
    ResNet50_model = ResNet50(weights='imagenet')
    
    ### Dog race model ###
    # Transfer learning
    print('Loading inception_resnet_v2')
    model_cropped = InceptionResNetV2(weights='imagenet', include_top=False)
    # End layers
    doggo_model = Sequential()
    # Input shape from the output of model_cropped using 224x224 images
    doggo_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 1536)))
    doggo_model.add(Dense(133))
    doggo_model.add(Dropout(0.4))
    doggo_model.add(Dense(133))
    doggo_model.add(Dropout(0.4))
    doggo_model.add(Dense(133, activation='softmax'))
    print('Loading end-layer weights')
    doggo_model.load_weights('models/weights.best.doggo_model.hdf5')

### Human detector
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index', methods=['POST'])
def index():
    
    return render_template('master.html')#, ids=ids, graphJSON=graphJSON)

@app.route('/predict' , methods=['POST'])
def predict():
    print("log: Got a predict call" , file=sys.stderr)
    
    # # Get image in byte format from request call
    byte_img = request.files['image'].read()
    # Transform bytes to image
    img = Image.open(BytesIO(byte_img))
    # Converts all to RGB format
    img = img.convert("RGB")
    # Resize to fit input of model
    img_resize = img.resize((224,224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img_resize)
    # 4d tensor
    tensor = np.expand_dims(x, axis=0)
    # Human face detection
    npimg = np.fromstring(byte_img, np.uint8)
    img_cv2 = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    human_yn = len(faces) > 0

    with session.graph.as_default():
        k.set_session(session)
        if human_yn:
            doggo_yn = False
        else:
            rn50_preprocess = rn50_preprocess_input(tensor)
            doggo_pred = np.argmax(ResNet50_model.predict(rn50_preprocess))
            # Imagenet labels https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
            doggo_yn = ((doggo_pred <= 268) & (doggo_pred >= 151))
        irnv2_preprocess = irnv2_preproccess_input(tensor)
        # Bottleneck features for dog race
        bottleneck = model_cropped.predict(irnv2_preprocess)
        # Dog race prediction
        race_pred = doggo_model.predict(bottleneck)
    
    race = dog_names[np.argmax(race_pred)]
    
    
    # Sending image back
    return_img = img
    rawBytes = io.BytesIO()
    return_img.save(rawBytes, "PNG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    
    return jsonify({'status':'success'},{'dog':str(doggo_yn)},{'human':str(human_yn)},{'race':race},{'dream':str(img_base64)})




# def mask_image():
	# # print(request.files , file=sys.stderr)
	# file = request.files['image'].read() ## byte file
	# npimg = np.fromstring(file, np.uint8)
    # #print(npimg.shape)
	# #img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	# ######### Do preprocessing here ################
	# # img[img > 150] = 0
	# ## any random stuff do here
	# ################################################
	# #img = Image.fromarray(img.astype("uint8"))
	# #rawBytes = io.BytesIO()
	# #img.save(rawBytes, "JPEG")
	# rawBytes.seek(0)
	# img_base64 = base64.b64encode(rawBytes.read())
	# return jsonify({'status':str(img_base64)})




# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()