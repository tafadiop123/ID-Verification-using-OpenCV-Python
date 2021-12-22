# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:59:30 2020

@author: a776942
"""
"""
1 - conda create -n darkflow-env python=3.8
2 - conda activate darkflow-env
3 - conda install cython numpy 
4 - conda config --add channels conda-forge
5 - conda install opencv
6 - cd darkflow
7 - python setup.py build_ext --inplace

"""
#A executer sur IPython
#!pip install opencv-python==4.1.1.26

# import os library 
import os 
os.chdir(r"C:\Users\a776942\Desktop\practise\darkflow") 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from darkflow.net.build import TFNet


# change the current directory 
# to specified directory 
#os.chdir(r"C:\Users\a776942\Desktop\practise\darkflow") 


options = {"model": r"C:\Users\a776942\Desktop\practise\darkflow\yolov2_custom.cfg",
           "load": -1,
           #"gpu": 1.0
           }

tfnet2 = TFNet(options)

tfnet2.load_from_ckpt()

""
def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
            
        
        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (0,255,0), 2)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        return newImage
                #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
	            #label_position = (x + int(w/2)), abs(y - 10)
	           #cv2.putText(img, result['label'], label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 2)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(original_img)
        
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(boxing(original_img, results))  
      
#path = r'C:\Users\a776942\Desktop\practise\CNI_PNG
#Lancement de l'itération sur les CNI
for filepath in glob.iglob(r'C:\Users\a776942\Desktop\practise\CNI_PNG\*.png'):
    original_img = cv2.imread(filepath, 1)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet2.return_predict(original_img)
    #original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    #print(results)
    
    #make a new folder
    croped_image = filepath.replace(".png","")
    if not os.path.exists(croped_image):
        os.makedirs(croped_image)
       
    
    for result in results:
        if result['label'] == 'cin' and result['confidence'] > 0.5:
            roi_cni =  original_img[result['topleft']['y']:result['bottomright']['y'],result['topleft']['x']:result['bottomright']['x']]
            roi_cni = cv2.cvtColor(roi_cni, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('roi CNI',roi_cni)
            cv2.imwrite(croped_image+'\cin.png',roi_cni)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
        if result['label'] == 'prenom':
            roi_prenom =  original_img[result['topleft']['y']:result['bottomright']['y'],result['topleft']['x']:result['bottomright']['x']]
            roi_prenom = cv2.cvtColor(roi_prenom, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('roi Prenom',roi_prenom)
            cv2.imwrite(croped_image+'\prenom.png',roi_prenom)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
        if result['label'] == 'nom' and result['confidence'] > 0.73:
            roi_nom =  original_img[result['topleft']['y']:result['bottomright']['y'],result['topleft']['x']:result['bottomright']['x']]
            roi_nom = cv2.cvtColor(roi_nom, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('roi Nom',roi_nom)
            #print(croped_image)
            cv2.imwrite(croped_image+'\name.png',roi_nom)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
        if result['label'] == 'photo':
            roi_photo =  original_img[result['topleft']['y']:result['bottomright']['y'],result['topleft']['x']:result['bottomright']['x']]
            roi_photo = cv2.cvtColor(roi_photo, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('roi',roi_photo)
            cv2.imwrite(croped_image+'\photo.png',roi_photo)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
           
    ####################################################################################################################
    ################ TAKE SELFIE #######################################################################################
                
    # Load HAAR classifier
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    # Load functions 
    def face_extractor(img):
        # function detects faces and returns the cropped face
        # If no face detcted, it returns the input image 
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if faces is ():
            return None
        
        #Crop all faces found
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+h]
        
        return cropped_face 
    
    #Initialize Webcam
    cap = cv2.VideoCapture(0)
    count = 0
    
    #Collect 100 samples of your face from Webcm input
    
    while True:
        ret,frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (712,635))
            face =cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Save file in specified direcory with unique name
            file_name_path = "./selfie/"+str(count)+".png"
            cv2.imwrite(file_name_path, face)
            
            #Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Face Cropper", face)
            
        else:
            print("Face not found")
            pass
        
        if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete !")       
    