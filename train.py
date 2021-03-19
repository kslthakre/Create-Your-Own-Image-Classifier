

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms,models
from torch import nn
from torch import optim
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import argparse



parser = argparse.ArgumentParser(description = 'Parser for train.py')

parser.add_argument('--data_dir',
                    action='store',
                    default='flowers',
                    type=str,
                    help='Input Dictionary')

parser.add_argument('--hidden_layer1',
                    action='store',
                    default=2048,
                    type=int,
                    help='1st layer')
					
parser.add_argument('--hidden_layer2',
                    action='store',
                    default=256,
                    type=int,
                    help='2nd layer')
					

parser.add_argument('--learning_rate',
                    action='store',
                    default=0.001,
                    type=float,
                    help='learning rate gradient descent')


parser.add_argument('--epochs',
                    action='store',
                    type=int,
                    default=3,
                    help='Define no of  epochs for training')

parser.add_argument('--gpu',
                    action='store',
                    dest='gpu',
                    default='cpu',
                    help='Use GPU for training')


parser.add_argument('--save-dir',
                    action='store',
                    dest='save_dir',
                    default='ImageClassifier',
                    type=str,
                    help='Set directory for the checkpoint, if not done all work will be lost')


parser.add_argument('--arch',
                    action='store',
                    default='vgg13',
                    help='Define which learning architectutre will be used')


in_args=parser.parse_args()
data_dir=in_args.data_dir
arch =in_args.arch
hidden_layer1 =in_args.hidden_layer1
hidden_layer2 =in_args.hidden_layer2
epochs =in_args.epochs  
learning_rate =in_args.learning_rate
gpu=in_args.gpu


data_dir = data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


print (train_dir)
print (valid_dir)
print (test_dir)

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

Valid_transform = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

valid_data = datasets.ImageFolder(valid_dir, transform=Valid_transform)



trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64,shuffle = True)

testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)

validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)



if arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    input = 25088
elif arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input = 25088

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
    else:
        device = torch.device("cpu")



# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input, hidden_layer1)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(hidden_layer2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)



model.to(device);

epochs = epochs
steps = 0
running_loss = 0
print_every = 48

# Implement a function for the validation pass

def validation(model, loader, criterion):
    test_loss = 0
    accuracy = 0
    
    for inputs1, labels1 in loader:
        
        inputs1, labels1 = inputs.to(device), labels.to(device)
        
        output1 = model.forward(inputs1)
        test_loss += criterion(output1, labels1).item()
        
        ps = torch.exp(output1)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            #valid_loss = 0
            #accuracy = 0
            model.eval()
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)
                    
                   
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

model.eval()
    
with torch.no_grad():
    testloss, accuracy = validation(model, testloader, criterion)
                
print(f"Test accuracy: {accuracy*100/len(testloader):.3f}%")			


			
			
model.class_to_idx = train_data.class_to_idx
checkpoint ={'model':model,
            'input_size': model.classifier[0].in_features,
            'output_size': 102,
            'learning_rate': learning_rate,
            'classifier': model.classifier,
            'epochs': epochs,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}
torch.save(checkpoint, 'project_checkpoint.pth')