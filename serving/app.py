from flask import Flask, jsonify
from torchvision import models
from i3d import InceptionI3d
from handler import *
import cv2
import numpy as np
import tempfile
from pathlib import Path
from keytotext import pipeline
from flask import request

app = Flask(__name__)

'''
request type: POST
description: process uploaded video and get prediction as response
request route: /predict
request params: file
response type: JSON
response: {
    data: [{frame_id : 23, time_stamp: 234, prediction: hello},
    {frame_id : 40, time_stamp: 237, prediction: how},
    {frame_id : 43, time_stamp: 239, prediction: are}]
    message: "success"
    status: "200 ok"
}'''

@app.route('/predict', methods=['POST'])
def predict():

    # load config
    res = []
    sentence = ""
    
    uploaded_file  = request.files['file']
    with tempfile.TemporaryDirectory() as td:
        temp_filename = Path(td) / 'uploaded_video'
        uploaded_file.save(temp_filename)
        print(str(temp_filename))
        res,sentence = load_model(file=str(temp_filename))
    print(res)

    # build response
    

    return jsonify({
        'data': [{'frame_id' : 23, 'time_stamp': 234, 'prediction': 'hello'},
        {'frame_id' : 40, 'time_stamp': 237, 'prediction': 'how'},
        {'frame_id' : 43, 'time_stamp': 239, 'prediction': 'are'}],
        'message': "success",
        'sentence': sentence,
        'status': "200 Ok"
})

