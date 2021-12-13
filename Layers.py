#bring in the custom L1 Distance layer as it is needed when loading the saved model

#import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

#build custom layer
class L1Dist(Layer):
    def __init__(self, **kwargs): #kwargs allows you to work on this as part of a bigger layer
        super().__init__() #inheritance
    
    #magic happens, the call tells this layer what to do when some data is passed
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
#perfoms the difference between the two passed layers(images)
