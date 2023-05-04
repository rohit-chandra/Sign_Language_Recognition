import yaml
from i3d import InceptionI3d
import cv2
import numpy as np
import tempfile
from pathlib import Path
from nslt_dataset_all import NSLT as Dataset
from keytotext import pipeline
from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import time


def load_conf_file(config_file):
    config = None
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_WLASL_dictionary(wlasl_class_list_file):
    
    global wlasl_dict 
    wlasl_dict = {}
    
    with open(wlasl_class_list_file) as file:
        for line in file:
            split_list = line.split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value
    return wlasl_dict

def load_model(file):
    # load config
    config = load_conf_file(config_file ='./config/config.yaml')

    frames = []
    words = []
    res = []
    sentence = ""
    offset = 0
    batch = 40
    text_count = 0
    text = " "
    text_list = []
    # initialize model
    global i3d
    i3d = InceptionI3d(config['model']['num_classes'], config['model']['in_channels'])
    i3d.load_state_dict(torch.load(config['model']['weights'], map_location=torch.device('cpu')))
    i3d.replace_logits(config['model']['num_classes'])
    i3d.eval()

    nlp_k2t = pipeline("k2t")

    vidcap = cv2.VideoCapture(str(file))
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print(video_length)
    ret, frame1 = vidcap.read()
    offset = offset + 1
    font = cv2.FONT_HERSHEY_TRIPLEX
    count = 0
    while vidcap.isOpened():

        if ret == True:
            count+=1
            offset+=1
            print(count, offset, offset % 20)
                        
            if offset > batch:
                w, h, c = frame1.shape
                sc = 224 / w
                sx = 224 / h
                frame = cv2.resize(frame1, dsize=(0, 0), fx=sx, fy=sc)
                frame1 = cv2.resize(frame1, dsize = (1280,720))
                frame = (frame / 255.) * 2 - 1

                frames.pop(0)
                frames.append(frame)
                    
                if offset % 20 == 0:
                    print("curr offset: ", offset)
                    text = run_on_tensor(torch.Tensor(np.asarray(frames, dtype=np.float32)))
                    if text != " ":
                        text_count = text_count + 1
                            
                        if bool(text_list) != False and bool(word_list) != False and text_list[-1] != text and word_list[-1] != text or bool(text_list) == False:
                            text_list.append(text)
                            word_list.append(text)
                            sentence = sentence + " " + text
                            
                        if(text_count > 2):
                            sentence = nlp(text_list,**params)
                            print("curr sentence percent 20: ", sentence)
                        cv2.putText(frame1, sentence, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)
                        res.append({'frame': frame1, 'prediction': text})
            else:
                w, h, c = frame1.shape
                sc = 224 / w
                sx = 224 / h
                frame = cv2.resize(frame1, dsize=(0, 0), fx=sx, fy=sc)
                frame1 = cv2.resize(frame1, dsize = (1280,720))
                frame = (frame / 255.) * 2 - 1

                frames.append(frame)
                if offset == batch:
                    print("was here")
                    text = run_on_tensor(torch.Tensor(np.asarray(frames, dtype=np.float32)))
                    print("the text: ", text)
                    if text != " ":
                        text_count = text_count + 1
                        if bool(text_list) != False and bool(word_list) != False and text_list[-1] != text and word_list[-1] != text or bool(text_list) == False:
                            text_list.append(text)
                            word_list.append(text)
                            sentence = sentence + " " + text

                                
                                        
                        if(text_count > 2):
                            sentence = nlp_k2t(text_list,**params)
                            print("curr sentence batch: ", sentence)

                        cv2.putText(frame1, sentence, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)
                        res.append({'frame': frame1, 'prediction': text})
                            
                    
                # if 0xFF == ord('q'):
                #     break
                        
            # cv2.putText(frame1, sentence, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)
                        
            if len(text_list) > 10:
                text_list.pop()
                text_list.pop()
                text_list.pop()
                res.pop()
                res.pop()
                res.pop()
            if (count > (video_length-1)):
                vidcap.release()
                break
    print("Done processing!") 
    return res,sentence
    


def run_on_tensor(ip_tensor):
    print("in fun on tensor", ip_tensor)

    ip_tensor = ip_tensor[None, :]
    
    t = ip_tensor.shape[2] 
    per_frame_logits = i3d(ip_tensor)
    print(per_frame_logits)
    predictions = F.upsample(per_frame_logits, t, mode='linear')

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    arr = predictions.cpu().detach().numpy()[0] 

    print(float(max(F.softmax(torch.from_numpy(arr[0]), dim=0))))
    print(wlasl_dict[out_labels[0][-1]])
    
    """
    
    The 0.5 is threshold value, it varies if the batch sizes are reduced.
    
    """
    if max(F.softmax(torch.from_numpy(arr[0]), dim=0)) > 0.5:
        return wlasl_dict[out_labels[0][-1]]
    else:
        return " " 

class PredictionResult():
    def __init__(self, frame_id=None, time_stamp=None, prediction=None):
        self.frame_id = frame_id
        self.time_stamp = time_stamp
        self.predict = predict
    


