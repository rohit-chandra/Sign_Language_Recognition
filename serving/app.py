from flask import Flask, jsonify



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

    ''' # load config
    config = load_conf_file('./config/config.yaml')
    # initialize model
    i3d = InceptionI3d(config.num_classes, config.in_channels=3)
    i3d.load_state_dict(torch.load(config.weights, map_location=torch.device('cpu')))
    i3d.replace_logits(num_classes)
    i3d.eval()
    # split video to frames
    uploaded_file  = request.files['file']
    with tempfile.TemporaryDirectory() as td:
            temp_filename = Path(td) / 'uploaded_video'
            uploaded_file.save(temp_filename)

            vidcap = cv2.VideoCapture(str(temp_filename))


    # process frames for i3d

    # predict words

    # generate sentence

    # build response
    '''

    return jsonify({
    'data': [{'frame_id' : 23, 'time_stamp': 234, 'prediction': 'hello'},
    {'frame_id' : 40, 'time_stamp': 237, 'prediction': 'how'},
    {'frame_id' : 43, 'time_stamp': 239, 'prediction': 'are'}],
    'message': "success",
    'sentence': "Hello, how are you?",
    'status': "200 ok"
})

