# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:57:57 2020

@author: user
"""


import PySimpleGUI as sg
import io, os
from PIL import Image
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from scipy import misc,ndimage
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


os.chdir('D:\\major project\\IMAGES\\')

def demo_photo_picker():
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,
                                   horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)


    training_set = train_datagen.flow_from_directory('Train_data',target_size = (64, 64),
                                                 batch_size = 32,class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('Test_data',target_size = (64, 64),
                                            batch_size = 32,class_mode = 'binary')
    classifier.fit_generator(training_set,
                         steps_per_epoch = 200,
                         nb_epoch = 1, 
                         validation_data = test_set,
                         validation_steps = 60)
    





    layout1 = [[
            sg.Image(key="image", size=(1300,150), filename="")
        ],
               [sg.Text("")],[sg.Text("")],[sg.Text("")],[sg.Text("")],
        [ 
            sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),
            sg.Text("Choose an image : ",font=("Roboto", 25)),
            sg.InputText(key="photo_file"),
            sg.FileBrowse(target="photo_file", file_types=(("JPEG Files","*.jpg"),( "PNG Files","*.png"),),size=(20,2))
        ], [sg.Text("")],[sg.Text("")],[sg.Text("")],[sg.Text("")], [
            sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    ")
            ,sg.Button(button_text="Show", button_color=("white", "springgreen4"), size= (20, 2)),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),
            sg.Button(button_text="Reset",button_color=("white", "blue"), size= (20, 2)), sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("      "),
            sg.Button(button_text="Exit", button_color=("white","firebrick3"), size= (20, 2))
            ]
    ]
    
    window = sg.Window('Pick a photo',size=(1366,768)).Layout(layout1)
    window3_active=False

    
    while True:
        
                
        event1, values = window.Read()
            
        if event1 is None or event1 == 'Exit':
            window.close()
            break
        if event1=="Reset":
            window.FindElement('photo_file').Update('')
            
        if event1 == "Show":
            if values["photo_file"] != "":
                #sg.Print("show button clicked")
                if os.path.isfile(values["photo_file"]):
                    extension = values["photo_file"].lower().split(".")[-1]
                    #print("file name: "+values["photo_file"])
                    new_filename = values["photo_file"].replace(extension, "png")
                    if extension in ["jpg", "jpeg", "jpe"]:  # JPG file
                        
                        im = Image.open(values["photo_file"])
                        im.save(new_filename)
                    layout2 = [[sg.Image(key="image", size=(1366,650), filename=new_filename)],
                           
                           [sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),
                            sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),
                            sg.Button(button_text="Predict", button_color=("white", "springgreen4"), size= (20, 2)),sg.Text("    "),
                            sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("      "),
                            sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("      "),sg.Text("    "),
                            sg.Button(button_text="Back", button_color=("white","firebrick3"), size= (20, 2))
                        ]
                        ]
                    window2=sg.Window("showing image",size=(1366,768)).Layout(layout2)
                    window.hide()
                    while True:
                        if window3_active==True:
                            window3_active=False
                            break
                        event2=window2.Read()[0]
                        if event2=='Back':
                            window.UnHide()
                            window2.close()
                            break
                        elif event2=='Predict':
                            sg.Print("prediction")
                            
                            test_image = image.load_img(new_filename, target_size = (64, 64))
                            test_image = image.img_to_array(test_image)


                            test_image = np.expand_dims(test_image, axis = 0)

                            result = classifier.predict(test_image)

                            #print(training_set.class_indices)
                            if result[0][0] == 1:
                                prediction = 'Malignant'
                            else:
                                prediction = 'Benign'
                            sg.Print(prediction)
                            #prediction = 'Benign'
                            layout3 = [[sg.Text(prediction, font=("Roboto", 40))],
                                       [sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),
                                        sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),
                                        sg.Button(button_text="Reset", button_color=("white", "springgreen4"), size= (20, 2)),sg.Text("    "),
                                        sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("      "),
                                        sg.Text("    "),sg.Text("    "),sg.Text("    "),sg.Text("      "),sg.Text("    "),
                                        sg.Button(button_text="Exit", button_color=("white","firebrick3"), size= (20, 2))
                                        ]
                                        ]
                            window3=sg.Window("output",size=(500,300)).Layout(layout3)
                            window2.hide()
                            while True:
                                event3=window3.Read()[0]
                                window3_active=True
                                if event3=='Reset':
                                    window2.close()
                                    window.UnHide()
                                    window.FindElement('photo_file').Update('')
                                    window3.close()
                                    break
                                elif event3=='Exit':
                                    window.close()
                                    window2.close()
                                    window3.close()
                                    return
    
                
                
                
                
                
                
    
                
            

if __name__ == "__main__":
    result = demo_photo_picker()

