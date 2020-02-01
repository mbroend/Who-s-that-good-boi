import json
import plotly
import numpy as np
import pandas as pd
import os , io , sys

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Violin, Box
from sklearn.externals import joblib
from sqlalchemy import create_engine
from io import BytesIO
import base64
#from keras import backend as K
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
from PIL import ImageFile, Image                           
ImageFile.LOAD_TRUNCATED_IMAGES = True    
#K.clear_session()
#ResNet50_model = ResNet50(weights='imagenet')
import tensorflow as tf

# def load_model():
	# global model
	# model = ResNet50(weights="imagenet")
            # # this is key : save the graph after loading the model
	# global graph
	# graph = tf.get_default_graph()
# load_model()

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def img_to_tensor(img):
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)   
    
def ResNet50_predict_labels(file):
    # returns prediction vector for image located at img_path
    img = rn50_preprocess_input(img_to_tensor(file))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(file):
    prediction = ResNet50_predict_labels(file)
    return ((prediction <= 268) & (prediction >= 151)) 

app = Flask(__name__)

# def tokenize(text):
    # tokens = word_tokenize(text)
    # lemmatizer = WordNetLemmatizer()

    # clean_tokens = []
    # for tok in tokens:
        # clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # clean_tokens.append(clean_tok)

    # return clean_tokens

# # load data
# engine = create_engine('sqlite:///../data/DisasterResponse.db')
# df = pd.read_sql_table('Messages', engine)

# # load model
# model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index', methods=['POST'])
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)
    
    # classes_count = df.drop(columns = ['id','message','genre']).sum()
    # classes_names = list(classes_count.index)
    
    # text_length = df['message'].str.len()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # graphs = [
        # {
            # 'data': [
                # Bar(
                    # x=genre_names,
                    # y=genre_counts
                # )
            # ],

            # 'layout': {
                # 'title': 'Distribution of Message Genres',
                # 'yaxis': {
                    # 'title': "Count"
                # },
                # 'xaxis': {
                    # 'title': "Genre"
                # },
                
            # }
        # },
        # {
            # 'data': [
                # Bar(
                    # x=classes_names,
                    # y=classes_count
                # )
            # ],

            # 'layout': {
                # 'title': 'Distribution of Classes in Training Data',
                # 'yaxis': {
                    # 'title': "Count"
                # },
                # 'xaxis': {
                    # 'title': "Classes"
                # }
            # }
        # },
        # {
            # 'data': [
                # Box(
                # x = text_length,
                # name = '' 
                # )
            # ],

            # 'layout': {
                # 'title': 'Box Plot of Text Length',
                # 'xaxis': {
                    # 'title': "Text length"
                # }
            # }
        # }
                
    # ]
    
    # # encode plotly graphs in JSON
    # ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    # graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html')#, ids=ids, graphJSON=graphJSON)

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


@app.route('/test' , methods=['GET','POST'])
def test():
    print("log: got at test" , file=sys.stderr)
    ResNet50_model = ResNet50(weights='imagenet')
    file = request.files['image'].read()
    #npimg = np.fromstring(file, np.uint8)
    #imgdata = file.split(',')[1]
    #decoded = base64.b64decode(imgdata)
    img = Image.open(BytesIO(file))
    img = img.resize((224,224))
    x = image.img_to_array(img)
    print(x.shape)
    # returns prediction vector for image located at img_path
    img = rn50_preprocess_input(np.expand_dims(x, axis=0))
    #print(img.shape)
    # with graph.as_default():
        # preds = model.predict(img)
    prediction = np.argmax(ResNet50_model.predict(img))
    #print(prediction)
    #prediction = ResNet50_predict_labels(np.argmax(ResNet50_model.predict(img)))
    print(((prediction <= 268) & (prediction >= 151)))
    
    
    model_cropped = InceptionResNetV2(weights='imagenet', include_top=False)
    bottleneck = model_cropped.predict(irnv2_preproccess_input(np.expand_dims(x, axis=0)))
    doggo_model = Sequential()
    doggo_model.add(GlobalAveragePooling2D(input_shape=bottleneck.shape[1:]))
    doggo_model.add(Dense(133))
    doggo_model.add(Dropout(0.4))
    doggo_model.add(Dense(133))
    doggo_model.add(Dropout(0.4))
    doggo_model.add(Dense(133, activation='softmax'))
    doggo_model.load_weights('models/weights.best.doggo_model.hdf5')
    prediction = doggo_model.predict(bottleneck)
    #race = dog_names[np.argmax(prediction)]
    print(np.argmax(prediction))
    
    
    
    #np.expand_dims(x, axis=0)
    
    #img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    #print(dog_detector(img))
    #print(file)
    #npimg = np.fromstring(file, np.uint8)
    #print(npimg)
    return jsonify({'status':'succces'})





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
    #app.run(host='127.0.0.1', port=3001, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()