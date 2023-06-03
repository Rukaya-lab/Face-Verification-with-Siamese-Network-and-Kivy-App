# Face-Verification-with-Siamese-Network-and-Kivy-App

### Siamese Neural Networks for One-shot Image Recognition
This work is on facial verification and recognition using siamese network and then building the app with Kivy.

Siamese Network model are good for task with small amount of data.

## Background
Facial recognition is a technology that can identify people by their faces. It can be used to capture a person's face from any image or video and then match it to a database of known faces. Facial recognition has become increasingly popular in recent years as a more secure and convenient way to authenticate users. It can replace traditional authentication methods such as passwords and one-time passwords.

Benefits of facial recognition:

- Security: Facial recognition is more secure than traditional authentication methods such as passwords and one-time passwords. This is because it is difficult to spoof someone's face.
- Convenience: Facial recognition is more convenient than traditional authentication methods. This is because users do not have to remember passwords or one-time passwords.
- Efficiency: Facial recognition can be used to quickly and efficiently authenticate users. This can save time and improve efficiency.

### Siamese Network
A Siamese network is a type of neural network (Convolutionall Ne) that consists of two or more identical networks that are trained together. The networks are trained to learn a shared representation of the input data, and they are then used to compare pairs of inputs and determine whether they are the same or different.
- The network would be trained on a dataset of images of faces. Each image would be paired with another image of the same person, or with an image of a different person. The network would then learn to identify the features that are common to all images of the same person, and to distinguish those features from the features of other people.
    - Once the network is trained, it can be used to compare two new images and determine whether they are of the same person. The network would do this by extracting features from each image and then comparing the features to see if they are similar.

Benefit:
- They are simple to train.
- They can learn complex relationships between data.
- They are suitable for small amount of data

### The Data

For this project there are two sets of Dataset.
One is a public data set of different labelled faces [Labelled Faces](http://vis-www.cs.umass.edu/lfw/), 
and the other dataset is of my faces at different angles and augmentations.


### Approach
- Since there are a lot of Images Processing, there is need to use GPU.
- The public dataset id labelled as the Negative dataset and my own images are bothe the positive data and the Anchor images.
    - open Cv libarary is used to collect the images from video capture.
    - The images collected were augmneted to create more samples of different properties to enable for a larger dataset.
    - All the collected images were resized to have a unifirm dimension of 100 by 100 
- The anchor images are then used with the positive and negative dataset to craete a paired and labeled dataset.
    - match a positive with an anchor and takes value of 1 since it the same person.
    - match negative and anchor and that will take a 0 since the two people wouldnt match.
    - That generated 6000 samples of paired data. 70% was for training and 30% for testing.        

#### The Model

##### Embedding Layer
- There are two embedding layer built. The Input embedding layer for the input images nad the Validation embedding layer for the validation images.
- Each model layer consist of:
    - The model consists of four convolution blocks, each followed by a max pooling layer. 
    - The convolution blocks use 64, 128, 128, and 256 filters, respectively. 
    - The max pooling layers use a pool size of 2x2 and a stride of 2. 
    - The final layer of the model is a dense layer with 4096 neurons. 
    - The activation function for all layers is ReLU, except for the final layer, which uses a sigmoid activation function.

### Distance Layer
A distance layer is built to calculate the L1 distance between two embeddings. 
- The L1 distance is a measure of the difference between two vectors. It is calculated by taking the absolute difference between the corresponding elements of the vectors.
- The distance is the the difference between the inpu embeddinga and the validation embedding.

** The Siamese network is then the amalgamtion of the Distance layers, embeddings and images. **

- The loss function is the Binary Cross Entropy.
- The optimizer is Adam.
- The metric is Accuracy, Precison, Recall
- The model achived a precision and recall of 1.0.

![](/images/verify.jpg)

### UI with Kivy
- Created a base UI app to interact with the model

![](/images/kivy.png)