#Import kivy Dependencies

#App layout
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

#Import other Depenencies
import cv2
import tensorflow as tf
import os
from Layers import L1Dist #from the layers file
import numpy as np

#Build app and layout
class CamApp(App):
    #define build function
    def build(self):
        #main app layout
        self.web_feed = Image(size_hint = (1, .8)) #image 
        self.button = Button(text= 'Verify', on_press = self.verify, size_hint = (1, .1)) #verify button
        self.verification_label = Label(text= 'Verification Uniniatiated', size_hint = (1, .1)) #image status

        #add item to layout
        layout = BoxLayout(orientation= 'vertical')
        layout.add_widget(self.web_feed)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #load keras model
        self.model = tf.keras.models.load_model('siamesemodelv1.h5', custom_objects= {'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
        #add capture from opencv
         #setup video capture
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    #update function to run continuously to get webcam feed
    def update(self, *args):
        #read frame
        ret, frame =self.capture.read()
        frame = frame[120:120+250, 200:200+250, :] #colon for all channels

        #flip image and convert to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        img_texture.blit_buffer(buf, colorfmt= 'bgr', bufferfmt = 'ubyte')
        self.web_feed.texture = img_texture

    #bring preprocess function
    def preprocess(self, file_path):
        #read the image from the path
        byte_img = tf.io.read_file(file_path)
        #load the image
        img = tf.io.decode_jpeg(byte_img)
    
        #resize the images to 100 *100
        img = tf.image.resize(img, (100,100))
        #scale
        img = img /255.0
    
        return img

    #bring over verification function
    #define function to build verification image and verify

    def verify(self, *args):

        #set threshold
        detection_threshold = 0.5
        verification_threshold = 0.8

        #capture input image and save to path
        save_path = (os.path.join('application_data', 'input_image', 'input_image.jpg'))
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(save_path, frame)


        #build result
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
        
            #make prediction
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        #detection threshold is metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        #verification threshold: proportion of positive predictions/total positve samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold
                                                           
        #set verification text
        self.verification_label.text = 'Person is Verified' if verified == True else 'Person is Not Verified'
        return results, verified

        #logout details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

if __name__ == '__main__':
    CamApp().run()    
