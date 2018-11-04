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
count=1

flag=1
pole=1






print("Content-type:text/html\r\n\r\n")
print("")
print ("<html>")

print ("</head>")


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
   
   open('video/'+f,'wb').write(fileitem.file.read())
   #message = 'The file "' + f + '" was uploaded successfully'
     
else:
   message = 'No file was uploaded'

uploaded_folder='video\\'+f

test_image = cv2.imread(uploaded_folder)



#############################################################################

#remove previous files!!
import glob
files = glob.glob('D:/Applications/XAMPP/htdocs/dashboard/final/video/data/*')
for f in files:
    os.remove(f)
   

##########################################################################





cc=0
cc1=0
cap = cv2.VideoCapture(uploaded_folder)

frameRate = cap.get(5) #frame rate
count = 0;
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    
    if (ret != True):
        break
    
    if ((frameId % 25.0) == 0):    
        name = 'D:/Applications/XAMPP/htdocs/dashboard/final/video/data/' + str(count) + '.jpg'
        
        cv2.imwrite(name, frame)
        count += 1
cap.release()













import glob
cv_img = []
for img in glob.glob("D:/Applications/XAMPP/htdocs/dashboard/final/video/data/*.jpg"):
    
    n= cv2.imread(img)
    n=cv2.resize(n,(128,128))
    
    
    
    
    n = np.array(n)
    n = n.astype('float32')
    n /= 225
    
    
    if K.image_dim_ordering()=='th':
        n=np.rollaxis(n,2,0)
        n= np.expand_dims(n, axis=0)
        
    else:
        n= np.expand_dims(n, axis=0)
       
		
#--------------------------------------------------------------------------------------------------------------        
if subject=="SEQUENTIAL" :
    loaded_model=load_model('model_seq.hdf5')
    
    cv_img = []
    for img in glob.glob("D:/Applications/XAMPP/htdocs/dashboard/final/video/data/*.jpg"):
        
        n= cv2.imread(img)
        n=cv2.resize(n,(128,128))
    
        
    
    
        n = np.array(n)
        n = n.astype('float32')
        n /= 225
        
    
        if K.image_dim_ordering()=='th':
            n=np.rollaxis(n,2,0)
            n= np.expand_dims(n, axis=0)
            
        else:
            n= np.expand_dims(n, axis=0)
            
    
    
    
    
        xx=loaded_model.predict(n,verbose=0) 
        yy=loaded_model.predict_classes(n)
        zz=(loaded_model.predict(n)[0][0])*100
        qq=(loaded_model.predict(n)[0][1])*100
        
        if(zz>15):
            cc+=1
    # print(("4",loaded_model.predict(n)))
   # print("5",loaded_model.predict_classes(n))
   #  print("6::::",(loaded_model.predict(n)[0][0])*100)
    # print("6::::",(loaded_model.predict(n)[0][1])*100)
    print(cc)
    score=(cc/count)*100
    print("<h1 style=" 'color:White;text-align:center;'">Video contains %.2f" %(score),  " % NSFW content</h1>" )
    
    if yy==0:
        flag=0
    
#--------------------------------------------------------------------------------------------------------------#  

elif subject=="U-NET" :
    loaded_model=load_model('model_unet.hdf5')
    
    cv_img = []
    for img in glob.glob("D:/Applications/XAMPP/htdocs/dashboard/final/video/data/*.jpg"):
        
        n= cv2.imread(img)
        n=cv2.resize(n,(128,128))
    
        
    
    
        n = np.array(n)
        n = n.astype('float32')
        n /= 225
        
    
        if K.image_dim_ordering()=='th':
            n=np.rollaxis(n,2,0)
            n= np.expand_dims(n, axis=0)
            
        else:
            n= np.expand_dims(n, axis=0)
        y_prob = loaded_model.predict(n,verbose=0) 
        y = y_prob.argmax(axis=-1)
        zz=(loaded_model.predict(n)[0][0])*100
        qq=(loaded_model.predict(n)[0][1])*100
        if(zz>15):
            cc+=1
   
    print(cc)
    score=(cc/count)*100
    print("<h1 style=" 'color:White;text-align:center;'">Video contains %.2f" %(score),  " % NSFW content</h1>" )
      
    if y==0:
        flag=0 
    
#--------------------------------------------------------------------------------------------------------------#  

elif subject=="SEQUENTIAL AND U-NET" :
    pole=0
    loaded_model=load_model('model_seq.hdf5')
    loaded_model1=load_model('model_unet.hdf5')
    
    cv_img = []
    for img in glob.glob("D:/Applications/XAMPP/htdocs/dashboard/final/video/data/*.jpg"):
        
        n= cv2.imread(img)
        n=cv2.resize(n,(128,128))
    
        
    
    
        n = np.array(n)
        n = n.astype('float32')
        n /= 225
        
    
        if K.image_dim_ordering()=='th':
            n=np.rollaxis(n,2,0)
            n= np.expand_dims(n, axis=0)
            
        else:
            n= np.expand_dims(n, axis=0)
            
    
    
    
    
        xx=loaded_model.predict(n,verbose=0) 
        y_prob = loaded_model1.predict(n,verbose=0)
        
        
        yy=loaded_model.predict_classes(n)
        y = y_prob.argmax(axis=-1)
        
        zz=(loaded_model.predict(n)[0][0])*100
        qq=(loaded_model.predict(n)[0][1])*100
        
        zz1=(loaded_model1.predict(n)[0][0])*100
        qq1=(loaded_model1.predict(n)[0][1])*100
        
        if(zz>15):
            cc+=1
            
        if(zz1>15):
            cc1+=1
  
    print(cc)
    print(cc1)
    score=(cc/count)*100
    print("<h1 style=" 'color:White;text-align:center;'">SEQUENTIAL PREDICTED Video contains %.2f" %(score),  " % NSFW content</h1>" )
  
    score=(cc1/count)*100
    print("<h1 style=" 'color:White;text-align:center;'">U-NET PREDICTED Video contains %.2f" %(score),  " % NSFW content</h1>" )
    if yy==0:
        flag=0
        
    if y==0:
        flag=0 
    ###############################################################################     
    """ 
    cc=0
    loaded_model=load_model('model_unet.hdf5')
    cv_img = []
    for img in glob.glob("D:/Applications/XAMPP/htdocs/dashboard/final/video/data/*.jpg"):
        
        n= cv2.imread(img)
        n=cv2.resize(n,(128,128))
        n = np.array(n)
        n = n.astype('float32')
        n /= 225
        
    
        if K.image_dim_ordering()=='th':
            n=np.rollaxis(n,2,0)
            n= np.expand_dims(n, axis=0)
            
        else:
            n= np.expand_dims(n, axis=0)
        
        y_prob = loaded_model.predict(n,verbose=0) 
        y = y_prob.argmax(axis=-1)
        zz=(loaded_model.predict(n)[0][0])*100
        qq=(loaded_model.predict(n)[0][1])*100
        
    if(zz>15):
        cc+=1
   
    print(cc)
    score=(cc/count)*100
    print("<h1 style=" 'color:White;text-align:center;'">U-NET PREDICTED Video contains %.2f" %(score),  " % NSFW content</h1>" )
        
    if y==0:
        flag=0   
    
  """  

    
#--------------------------------------------------------------------------------------------------------------#  

else :
    print("<div><h2 style=" 'color:Red;text-align:center;'">PLEASE TRY AGAIN!!!</h2></div>")
    flag=2

#--------------------------------------------------------------------------------------------------------------#  

########################################################################
print("<div>")   
if flag==1 : 
    print("<video width="'320'" height="'240'" style='display: block; margin-left: auto; margin-right: auto;' controls>")
    print("<source src=%s type="'video/mp4'" >"%uploaded_folder)
    print("</video>") 
else :
    print("<h2 style=" 'color:red;text-align:center;'"><b>Video CANNOT BE DISPLAYED DUE TO PRESENCE OF NSFW CONTENT</h2>")
print("</div>")       

"""
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
"""
#print ("</pre>")
print("</body>")
print("</html>")
