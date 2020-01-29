import json
import plotly
import pandas as pd
import os , io , sys

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Violin, Box
from sklearn.externals import joblib
from sqlalchemy import create_engine


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

def mask_image():
	# print(request.files , file=sys.stderr)
	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	######### Do preprocessing here ################
	# img[img > 150] = 0
	## any random stuff do here
	################################################
	img = Image.fromarray(img.astype("uint8"))
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	img_base64 = base64.b64encode(rawBytes.read())
	return jsonify({'status':str(img_base64)})


@app.route('/test' , methods=['GET','POST'])
def test():
    print("log: got at test" , file=sys.stderr)
    file = request.files['image'].read()
    print(file)
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