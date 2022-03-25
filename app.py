
# import os
from flask import Flask
from flask import render_template
from flask import request
from flask_bootstrap import Bootstrap
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import glob
# import shutil

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader,Dataset
# from torchvision.models import resnet18, resnet34 ,resnet50
import os
import torchvision.models as models




app = Flask(__name__)
# UPLOAD_FOLDER = r"C:\Users\Salio\PycharmProjects\Project\static\images"
UPLOAD_FOLDER = "/usr/src/app/static/images"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_gpu = torch.cuda.is_available()

net = models.mobilenet_v2(pretrained=True)
# net = torch.load(checkpoint_fpath)
# net.fc = nn.Linear(1280,7)
net.classifier[1] = nn.Linear(1280, 7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001,
                      momentum=0.9, weight_decay=5e-4)


def load_ckp(checkpoint_fpath, model, optimizer):
    # load check point
    if torch.cuda.is_available() == False:

      checkpoint = torch.load(checkpoint_fpath,map_location ='cpu')
    else:
        checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model--->checkpoint = { 'state_dict': model.state_dict() }
    # model.load_state_dict(torch.load('model_weights.pth'))
    model.load_state_dict(checkpoint['state_dict'])  # --> model.load_state_dict(model.state_dict())
    # initialize optimizer from checkpoint to optimizer--->checkpoint = { 'optimizer': optimizer.state_dict()}
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
# ckp_path = r"C:\Users\Salio\PycharmProjects\Project\static\best_st3.pt."
ckp_path = "/usr/src/app/static/best_st3.pt"
net, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, net, optimizer)
# print("model = ", net)
# print("optimizer = ", optimizer)
# print("start_epoch = ", start_epoch)
# # print("valid_loss_min = ", valid_loss_min)
# print("valid_loss_min = {:.6f}".format(valid_loss_min))

import torchvision.transforms as transforms
from PIL import Image


def pre_image_new(image_path, model):
    # img_path = 'static/0c69e6c4-3783-4b88-a451-95c91c411821.jpg'
    img = Image.open(image_path)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std)])
    # get normalized image
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    # print(img_normalized.shape)
    # input = Variable(image_tensor)
    # img_normalized = img_normalized.to(device)
    # print(img_normalized.shape)

    with torch.no_grad():
        model.eval()

        output = model(img_normalized)
        # print(output)
        index = output.data.cpu().numpy().argmax()
        # print(index)
        cla = ['MoreThanSeventeenDays', 'NoBruise', 'SeventeenDays', 'SixDays', 'ThreeDays', 'TwelveDays', 'TwoDays']
        NameClass = cla[index]

        return NameClass

# index1 = pre_image_new(r"C:\Users\Salio\PycharmProjects\app\static\012d4145-ecba-4e36-91fc-47980f2614d5.jpg",net)
# print("this###########")
# print(index1)
# a = index1

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = pre_image_new(image_location,net)
            return render_template('index.html', prediction=pred,image_loc =image_file.filename )
    return render_template('index.html',prediction = 0,image_Loc = None)

if __name__ =="__main__":
    app.run()