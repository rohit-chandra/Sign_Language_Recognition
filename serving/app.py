from flask import Flask, jsonify, render_template
from handler import *
from flask import request
import os
# from camera import VideoCamera



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
    status: "200"
}'''
class APIError(Exception):
    """All custom API Exceptions"""
    pass

class BadRequest(APIError):
    code = 400
    description = 'Bad request'

@app.errorhandler(APIError)
def handle_exception(err):
    """Return custom JSON when APIError or its children are raised"""
    response = {"error": err.description, "message": "", "status": err.code}
    if len(err.args) > 0:
        response["message"] = err.args[0]
    return jsonify(response)

@app.errorhandler(500)
def handle_exception(err):
    """Return JSON instead of HTML for any other server error"""
    response = {"error": "Sorry, that error is on us, please contact support if this wasn't an accident","status": err.code}
    return jsonify(response)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():

    res = []
    sentence = ""

    if 'file' in request.files:
        uploaded_file  = request.files['file']
        uploaded_file.save(os.path.join('data', uploaded_file.filename))
        res,sentence = load_model(file=os.path.join('data', uploaded_file.filename))

        return jsonify({
            'data': res,
            'message': "success",
            'sentence': sentence,
            'status': "200"
            })
    else:
        raise BadRequest("File missing or invalid, please upload a .mp4 file")
      

def gen(camera):
    while True:
        try:
            frame = camera.get_frame()
            print(frame)
        except Exception:
            print("Video is finished or empty")
            # return None
            frame = camera.get_heartbeat()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():

    res = []
    sentence = ""

    try:
    
        res,sentence = load_model_for_live()

        return jsonify({
            'data': res,
            'message': "success",
            'sentence': sentence,
            'status': "200"
            })
    except:
        raise BadRequest("Give Video permissions")
    # return Response(gen(VideoCamera()),
    #                 mimetype='multipart/x-mixed-replace; boundary=frame')

