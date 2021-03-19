import matplotlib.pyplot as plt
import torch
import PIL
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
import numpy as np
import seaborn as sns
from collections import OrderedDict
import argparse


parser=argparse.ArgumentParser(description='for Predict.py:')
parser.add_argument('--image',
                    action='store',
                    default='flowers/test/1/image_06743.jpg',
                    help='input image path')

parser.add_argument('--checkpoint',
                    action='store',
                    default='./project_checkpoint.pth',
                    help='load saved checkpoint')

parser.add_argument('--top_k',
                    action='store',
                    default=5,
                    type=int,
                    help='te get top 5 classes')

parser.add_argument('--category_names',
                    default='cat_to_name.json',
                    help='to get the names')

parser.add_argument('--gpu',
                    action='store',
                    default='cpu',
                    help='if gpu avalible then user can opt cuda taking default to cpu')

in_args=parser.parse_args()
image =in_args.image
checkpoint =in_args.checkpoint
top_k =in_args.top_k  
category_names=in_args.category_names
json_file = in_args.category_names
gpu = in_args.gpu

with open(json_file, 'r') as f:
    cat_to_name = json.load(f)
                    


def load_checkpoint(path):
    checkpoint = torch.load(in_args.checkpoint)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
model = load_checkpoint(in_args.checkpoint)
model

model = load_checkpoint('checkpoint')

model = model.eval()

from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    #Ref :https://knowledge.udacity.com/questions/183400
    

    im = Image.open(image)

   
    width, height = im.size

    aspect_ratio = width / height
    if width < height: resize_size=[256, 256**1200]
    else: resize_size=[256**1200, 256]
        
    im.thumbnail(size=resize_size)

    
   
    left = (width - 224)/12
    top = (height - 224)/12
    right = left + 224
    bottom = top + 224
    im = im.crop((left, top, right, bottom))
    

   
    np_image = np.array(im)/255 

    
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
 
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image


def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    
    
    model.to('cpu')
    
    # Set model to evaluate
    model.eval();

   
    processed_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to('cpu')
    outputs  = model.forward(processed_image)
    outputs  = torch.exp(outputs)
    top_p, top_class  = outputs.topk(top_k)
    top_p = np.array(top_p.detach())[0] 
    top_class = np.array(top_class.detach())[0]
    # https://knowledge.udacity.com/questions/386740
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_class = [idx_to_class[lab] for lab in top_class]
    top_flowers = [cat_to_name[lab] for lab in top_class]
    
    return top_p, top_class, top_flowers
	
	
	
image_path = image

probability, classes,Flower_names = predict(image_path, model,top_k)
print(probability*100)  
print(Flower_names)