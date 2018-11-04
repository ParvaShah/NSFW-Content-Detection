#!C:/Users/PARVA SHAH/Anaconda3/python.exe

import cgi, os
import cgitb; cgitb.enable()


import cv2
import numpy as np


from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltt; pltt.rcdefaults()
import matplotlib.pyplot as pltt

from keras import backend as K
K.set_image_dim_ordering('th')


flag=1
pole=1
print("Content-type:text/html\r\n\r\n")
print("")
print ("<html>")

print ("<body style="'background-color:Black; text-align: center;'">")
form = cgi.FieldStorage()

if form.getvalue('subject'):
   subject = form.getvalue('subject')
else:
   subject = "NO MODEL SELECTED"

if subject=="BOTH":
    subject="SEQUENTIAL AND U-NET"

if subject!="NO MODEL SELECTED" :
    print("<h1 style=" 'color:Lime;text-align:center;'"> YOU HAVE SELECTED %s MODEL</h1>" % (subject))

else :
    print("<h1 style=" 'color:Red;text-align:center;'"> NO MODEL HAS BEEN SELECTED </h1>")
fileitem = form['filename']

# Test if the file was uploaded
if fileitem.filename:
   # strip leading path from file name to avoid 
   # directory traversal attacks
   f = os.path.basename(fileitem.filename)
   
   open('files/'+f,'wb').write(fileitem.file.read())
   #message = 'The file "' + f + '" was uploaded successfully'
     
else:
   message = 'No file was uploaded'

uploaded_folder='files/'+f





test_image = cv2.imread(uploaded_folder)


test_image=cv2.resize(test_image,(128,128))


test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
num_channel=1

if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		
else:
		test_image= np.expand_dims(test_image, axis=0)
		
#--------------------------------------------------------------------------------------------------------------#        
if subject=="SEQUENTIAL" :
    loaded_model=load_model('model_seq.hdf5')
    
    ###############################################################################
    xx=loaded_model.predict(test_image,verbose=0)
    objects=('NSFW','Not NSFW')
    y_pos=np.arange(len(objects))
    prediction=[xx[0][0]*100,xx[0][1]*100]
    
    
    plt.barh(y_pos,prediction,align='center',alpha=0.3)
    plt.yticks(y_pos,objects)
    plt.xlabel('PREDICTION')
    plt.title('NSFW CONTENT DETECTION [SEQUENTIAL]')
    plt.savefig('predictgraph/predictiongraph'+f.split('.')[0]+'.jpg')
  #################################################################################
  
    y=loaded_model.predict_classes(test_image,verbose=0)
    print("<div>")
    if y==0 :
        print("<h1 style=" 'color:White;text-align:center;'">Image contains %.2f" %((loaded_model.predict(test_image)[0][0])*100),"% NSFW content</h1>")
        flag=0
    else:
        print("<h1 style=" 'color:White;text-align:center;'">Image contains %.2f" %(100.00-(loaded_model.predict(test_image)[0][1])*100),"% NSFW content</h1>")
   
    
    print("</div>")

#--------------------------------------------------------------------------------------------------------------#  

elif subject=="U-NET" :
    loaded_model=load_model('model_unet.hdf5')
    
     ###############################################################################
    xx=loaded_model.predict(test_image,verbose=0)
    objects=('NSFW','Not NSFW')
    y_pos=np.arange(len(objects))
    prediction=[xx[0][0]*100,xx[0][1]*100]
    
    
    plt.barh(y_pos,prediction,align='center',alpha=0.3)
    plt.yticks(y_pos,objects)
    plt.xlabel('PREDICTION')
    plt.title('NSFW CONTENT DETECTION [U-NET]')
    plt.savefig('predictgraph/predictiongraph'+f.split('.')[0]+'.jpg')
  #################################################################################
    
    y_prob = loaded_model.predict(test_image) 
    y = y_prob.argmax(axis=-1)
    print("<div>")
    if y==0 :
        print("<h1 style=" 'color:White;text-align:center;'">Image contains %.2f" %((loaded_model.predict(test_image)[0][0])*100),"% NSFW content</h1>")
        flag=0
    else:
        print("<h1 style=" 'color:White;text-align:center;'">Image contains %.2f" %(100.00-(loaded_model.predict(test_image)[0][1])*100),"% NSFW content</h1>")
    
    
     
    print("</div>")
    
#--------------------------------------------------------------------------------------------------------------#  

elif subject=="SEQUENTIAL AND U-NET" :
    pole=0
    loaded_model=load_model('model_seq.hdf5')
    
     ###############################################################################
    xx=loaded_model.predict(test_image,verbose=0)
    objects=('NSFW','Not NSFW')
    y_pos=np.arange(len(objects))
    prediction=np.array([xx[0][0]*100,xx[0][1]*100])
    
    
    
    plt.barh(y_pos,prediction)
    plt.yticks(y_pos,objects)
    plt.xlabel('PREDICTION')
    plt.title('[SEQUENTIAL]--NSFW CONTENT DETECTION--[U-NET]')
    #################################################################################
  
    y=loaded_model.predict_classes(test_image,verbose=0)
    print("<div>")
    print("<h1 style=" 'color:White;text-align:center;'">SEQUENTIAL predicted ")
    
    
    if y==0 :
        print("Image contains %.2f" %((loaded_model.predict(test_image)[0][0])*100),"% NSFW content</h1>")
        flag=0
    else:
        print("Image contains %.2f" %(100.00-(loaded_model.predict(test_image)[0][1])*100),"% NSFW content</h1>")
    
    loaded_model=load_model('model_unet.hdf5')
    
     ###############################################################################
    xxx=loaded_model.predict(test_image,verbose=0)
 
    predictionn=np.array([xxx[0][0]*100,xxx[0][1]*100])
    
    
    plt.barh(y_pos,-predictionn)
    plt.yticks(y_pos,objects)
    
    plt.savefig('predictgraph/c'+f.split('.')[0]+'.jpg')
   #################################################################################
  
    y_prob = loaded_model.predict(test_image) 
    z = y_prob.argmax(axis=-1)
    
    print("<h1 style=" 'color:White;text-align:center;'">U-NET predicted ")
    if z==0 :
        print("Image contains %.2f" %((loaded_model.predict(test_image)[0][0])*100),"% NSFW content</h1>")
        flag=0
    else:
        print("Image contains %.2f" %(100.00-(loaded_model.predict(test_image)[0][1])*100),"% NSFW content</h1>")
    
    print("</div>")
    
#--------------------------------------------------------------------------------------------------------------#  

else :
    print("<div><h2 style=" 'color:Red;text-align:center;'">PLEASE TRY AGAIN!!!</h2></div>")
    flag=2

#--------------------------------------------------------------------------------------------------------------#  

########################################################################
print("<div>")    

#only one model and NSFW
if (flag==0 and pole==1) :
    print("<h2 style=" 'color:red;text-align:center;'"><b>IMAGE CANNOT BE DISPLAYED DUE TO PRESENCE OF NSFW CONTENT</h2>")
    print("<img src='%s' alt='GRAPH OF PREDICTION' width='800' height='400' style= 'display: block; margin-left: auto; margin-right: auto;'>" %('predictgraph/predictiongraph'+f.split('.')[0]+'.jpg'))

#both models and NSFW
elif (flag==0 and pole==0) :
    print("<h2 style=" ' color:red;text-align:center;'"><b>IMAGE CANNOT BE DISPLAYED DUE TO PRESENCE OF NSFW CONTENT</h2>")
   # print("<img src='%s' alt='GRAPH OF PREDICTION' width='600' height='400'>" %('predictgraph/predictiongraph'+f.split('.')[0]+'.jpg'))
    print("<img src='%s' alt='GRAPH OF PREDICTION1' width='800' height='400' style='display: block; margin-left: auto; margin-right: auto;'>" %('predictgraph/c'+f.split('.')[0]+'.jpg'))


#only one model and not NSFW
elif(flag==1 and pole==1):
    print("<div>")
    print("<img src='%s' alt='IMAGE THAT YOU UPLOADED' width='700' height='400' style=' margin-left: 10px;'>" %(uploaded_folder))
    print("<img src='%s' alt='GRAPH OF PREDICTION' width='700' height='400' style=' margin-left: 85px;'>" %('predictgraph/predictiongraph'+f.split('.')[0]+'.jpg'))
    print("</div>")
#both models and not NSFW
elif (flag==1 and pole==0):
    print("<img src='%s' alt='IMAGE THAT YOU UPLOADED' width='700' height='400' style='margin-left: 10px;'>" %(uploaded_folder))
   # print("<img src='%s' alt='GRAPH OF PREDICTION' width='600' height='400'>" %('predictgraph/predictiongraph'+f.split('.')[0]+'.jpg'))
    print("<img src='%s' alt='GRAPH OF PREDICTION1' width='700' height='400' style=' margin-left: 87px;'>" %('predictgraph/c'+f.split('.')[0]+'.jpg'))


else :
    print("<h3 style=" 'text-align:center;'"><b>IMAGE CANNOT BE DISPLAYED AS ANY OF THE PROVIDED MODEL IS NOT SELECTED</h3>")


print("</div>")    
######################################################################

print ("</pre>")
print ("</body>")

print ("</html>")