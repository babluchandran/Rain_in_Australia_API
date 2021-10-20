import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    features = [np.array(list(data.values()))]
    prediction = model.predict(predict_input(features))
   
    return jsonify(prediction[0])
    
def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    pred = model.predict(input_df)[0]
    return pred

if __name__ == "__main__":
    app.run(port=5000, debug=True)