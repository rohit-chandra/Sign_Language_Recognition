import requests
from flask import jsonify
import json
import yaml


def load_conf_file(config_file):
    config = None
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

global config
config = load_conf_file(config_file ='./config/config.yaml')

def k2t_query(payload):
    ip_payload = {
        "inputs": payload
    }
    token_ = config['model']['hugging_face_k2t_token']
    headers = {"Authorization": f"Bearer {token_}"}
    print(type(ip_payload))
    # ip_payload = json.loads(json.dumps(ip_payload))
    try: 
        response = requests.post(config['model']['hugging_face_k2t_api'], headers=headers, json=ip_payload)
        print(response)
        resp_obj = response.json()
        print("here",resp_obj[0]['generated_text'])
        sentence = ""
        return sentence
    except:
        return ""
    
k2t_query("hi how are you")