from PIL import Image
import numpy as np
import sys
import os
import csv
#new calls
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

#Useful function
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
myFileList = createFileList('C:/Users/andre/OneDrive/Documents/APS360/dataset/VIL100/JPEGImages/0_Road001_Trim003_frames')

for file in myFileList:
    print(file)
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()
	
	#Heres the new stuff
	#convert image to tensor
    newimg = T.ToTensor()(img_grey)
    newimg = newimg.unsqueeze(0)
	#apply max pool to reduce by 64x
    pooling = nn.MaxPool2d(8,8,0)
    newimg = pooling(newimg)
    newimg = newimg.squeeze(0)
	#convert back to image
    newimg = T.ToPILImage()(newimg)

    # Save Greyscale values
    value = np.asarray(newimg.getdata(), dtype=np.int).reshape((newimg.size[1], newimg.size[0]))
    value = value.flatten()
    print(value)
	
    with open("new_img.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)