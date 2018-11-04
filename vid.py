# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 18:45:45 2018

@author: PARVA SHAH
"""

import cv2


videoFile = "nsfw1.mp4"
imagesFolder = "/data"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
count = 0;
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
   
    if (ret != True):
        break
    
    if ((frameId % 25.0) == 0):    
        name = './data/frame' + str(count) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        count += 1
cap.release()
print ("Done!",count)

#------------------------------------------------------------------



from keras.models import load_model

import os
import cv2
import numpy as np



from keras import backend as K

K.set_image_dim_ordering('th')





loaded_model=load_model('model2.hdf5')



cc=0


import glob
cv_img = []
for img in glob.glob("D:\D Python Programs\check1\data\*.jpg"):
    print(img)
    n= cv2.imread(img)
    n=cv2.resize(n,(128,128))
    
    
    
    
    n = np.array(n)
    n = n.astype('float32')
    n /= 225
    print (n.shape)
    
    if K.image_dim_ordering()=='th':
        n=np.rollaxis(n,2,0)
        n= np.expand_dims(n, axis=0)
        print (n.shape)
    else:
        n= np.expand_dims(n, axis=0)
        print (n.shape)
    
   # print(("4",loaded_model.predict(n)))
   # print("5",loaded_model.predict_classes(n))
   #  print("6::::",(loaded_model.predict(n)[0][0])*100)
    # print("6::::",(loaded_model.predict(n)[0][1])*100)
    
    y_prob = loaded_model.predict(n) 
    y = y_prob.argmax(axis=-1)
    zz=(loaded_model.predict(n)[0][0])*100
    xx=(loaded_model.predict(n)[0][1])*100
    print("4",y_prob)
    print("5",y)
    print("6::::",zz)
    print("6::::",xx)
    if(zz>15) :
        cc+=1
score=(cc/count)*100
print("Video contains %.2f % NSFW content" %(score,"%"))
print("COMPLETE")













