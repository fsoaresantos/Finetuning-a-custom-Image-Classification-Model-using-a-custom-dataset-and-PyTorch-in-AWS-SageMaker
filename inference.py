#Import dependencies.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms

import time
import os
import shutil
import json
import logging
from PIL import Image
import io
import pickle

rlog = logging.getLogger()
rlog.setLevel(logging.DEBUG)
handler = logging.FileHandler('inf_logging.log') ## define logging file
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
rlog.addHandler(handler)
###rlog.setLevel(logging.DEBUG)


def net():
    '''
        This function initializes a pretrained model.
    '''

    # download pretrained model
    model = models.resnet18(pretrained = True)
    
    # freeze all the pre-trained layers
    for p in model.parameters():
        p.requires_grad = False
    
    # get the numbers of input features in the last FC layer
    nin_features = model.fc.in_features
    
    # replace the last FC layer to meet the number of classes (out_features) in the actual dataset
    model.fc = nn.Sequential(nn.Linear(in_features = nin_features, out_features = 133)) #oss. there're 133 n_classes in the dataset
    
    return model



def model_fn(model_dir):
    """
    Load/deserialize the model and return it
    
    Args:
    model_dir -- the directory path to model artifacts
    """

    ##device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ###MDed::::
    
    ## timing the loading of the model
    print("-"*20+"loading model"+"-"*20, flush=True)
    s_t = time.perf_counter()
    
    model = net()
    
    modeldir_path = os.path.join(model_dir, "model.pth")
    
    ## load the model
    model.load_state_dict(torch.load(modeldir_path))
    #with open(modeldir_path, 'rb') as f:
    #model.load_state_dict(torch.load(f))
    
    f_t = time.perf_counter()
    print(f"loading time: {f_t - s_t} seconds")
    print("-"*64)

    rlog.info("at model_fn: Model loaded!") ###::LOG
    
    ##return model.to(device) ###MDed::::!!! sent to device without define device!!!
    return model
    


def input_fn(request_body, request_content_type):
    """
    Deseriaize request_body into an object that can be used for prediction
    
    Args:
    request_body is a path to an image (as 'bytearray' object)
    request_body is a PIL image (as 'bytearray' object)
    
    Return: Pill.Image object
    
    """
    rlog.info(f"at input_fn: Request content type {request_content_type}") ###::LOG
    
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if request_content_type == "image/jpeg":
        rlog.info(f"at input_fn: input body type {type(request_body)}") ###::LOG
        request = transform(Image.open(io.BytesIO(request_body)))
        rlog.info(f"at input_fn: return type {type(request)}") ###::LOG
        return(request)
    
    raise Exception(f"Invalid content type in input_fn: entered {request_content_type}, requested 'image/jpeg'")



def predict_fn(input_object, model):
    """
    Apply the model to the deserialized object
    input_object is the object returned by input_fn
    model is the model loaded by model_fn
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_object = input_object.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        #output = model(input_object)
        output = model(input_object.unsqueeze(0))
            
    return output.numpy()



