# Image-Classification-on-the-ASL-Alphabet-Dataset-Utilizing-Convolutional-Neural-Networks




# <a id = 'ProblemStatement'>Problem Statement</b></a>

Deaf and hard-of-hearing communities deserve to have a voice in, and be heard by, institutions and decision-makers. Barriers to full inclusion in educational, economic, and civic life persist due to lack of attention paid to, or resources allocated for, sign language translation services.

The objective of this project is to develop a tool that will translate fingerspelling in the American Sign Language (ASL) alphabet into text. The project will use image data as inputs for a Convolutional Neural Network (CNN) to develop an algorithm to classify 29 possible outcomes, using the model's accuracy score (F1) to determine which to deploy on a sign language video with the goal of translating ASL letters into text.

Research questions include:
- How accurately can a CNN model classify the 29 ASL classes, including the letters A-Z, space, delete, and no sign?
- How accurately can a model trained on this dataset classify frames from a video with ASL fingerspelling (aka testing the model using external data)?


---

# <a id = 'Content'> Content </b></a>
- [Executive Summary](#ExecutiveSummary)
- [Repo Structure](#RepoStructure)    
- [Library Requirements](#LibraryRequirements)
- [Dataset Description](#DataSetDescription)
- [1. Image Processing](#ImageProcessing)
- [2. Modeling Methodology](#ModelingMethodology)
    - [2.1 Overview](#Overview)
    - [2.2 Convolutional Neural Networks (CNNs/ConvNets)](#CNNs)
- [Results and Discussion](#Results)    
- [Recommendations](#Recommendations)

---

# <a id = 'ExecutiveSummary'>Executive Summary</b></a>
Language translation presents a unique challenge to data scientists interested in natural language processing. Languages are many and complex, but the benefits of effective translation tools for cross-cultural communication in a globalized economy cannot be overstated. Efforts to use computers to translate language in the United States began in the 1950s with written text, and have evolved over time as computational power increased, approaches to statistical modeling became more advanced, and access to natural language data multiplied.

Technologies for translating sign language lag those for written and spoken languages. Sign language translation is especially challenging because it is communicated through physical movements and expression, and is processed visually. Sign language involves movements of the hands, arms, head, shoulders, and torso. Facial expressions are also important for communication using sign language. Adding to the complexity, sign language is not a universal language, rather different countries have their own sign language and different regions have unique dialects. As a results of these translation challenges, Deaf and hard-of-hearing communities around the world are systematically underserved by advances in technology for translation.


---
# <a id = 'RepoStructure'> Repo Structure </b></a>
## Code/ <br />

*Image Processing and models:*\
&nbsp; &nbsp; &nbsp; __ [MainCode.ipynb](/Code/MainCode.ipynb)<br />

## asl_alphabet_train/<br />
*In order to successfully compile the code and models, you would need to download the ALS Alphabet image dataset from Kaggle using the folloiwng link and put them inside the **asl_alphabet_train** directory:*\
&nbsp; &nbsp; &nbsp; __ [Link_asl_alphabet_train](https://www.kaggle.com/grassknoted/asl-alphabet)<br />


[README.md](README.md)<br />



---
# <a id = 'LibraryRequirements'>Library Requirements</b></a>

- Tensorflow
- Tensorflow-io
- opencv

---

# <a id = 'DataSetDescription'>Dataset Description</b></a>

American Sign Language (ASL) is the primary form of sign language for English-speaking people who are Deaf or hard-of-hearing in the United States and Canada. The data set is from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet), and contains a collection of images of the American Sign Language alphabet, separated in 29 folders which represent the classes. The training data set contains 87,000 images total, each of which are 200x200 pixels. There are 3,000 images for each of the 29 classes. The letters A-Z comprise 26 of the classes, and there are 3 additional classes for "space", "delete" and "nothing".

---


# <a id = 'ImageProcessing'>Image Processing</b></a>
Before performing any task related to images, it is almost always necessary to first process the images to make them more suitable as input data.<br>
What is Preprocessing? Augmentation? **Image preprocessing** are the steps taken to format images before they are used by the model training and inference. This includes, but is not limited to, resizing, orienting, and color corrections. **Image augmentation** are manipulations applied to images to create different versions of similar content in order to expose the model to a wider array of training examples. For example, randomly altering rotation, brightness, or scale of an input image requires that a model consider what an image subject looks like in a variety of situations.<br>
Why Preprocess Data? Preprocessing is required to clean image data for model input. For example, fully connected layers in convolutional neural networks required that all images are the same sized arrays. Image preprocessing may also decrease model training time and increase model inference speed. If input images are particularly large, reducing the size of these images will dramatically improve model training time without significantly reducing model performance.([source](https://blog.roboflow.com/why-preprocess-augment/))<br>
In this project images were converted from JPEG or PNG files to a usable data for our neural networks. The Library used was the TensorFlow 2.0 as it provides a variety of utility functions to obtain image data from files, resize the images and transform a large set of images all at once. TensorFlow I/O is a collection of file systems and file formats that are not available in TensorFlow's built-in support. It provides useful extra Dataset, streaming, and file system extensions. The tensorflow-io package provides a list of color space conversions APIs that can be used to prepare and augment the image data.
Preprocessing steps used for this project were:

- Image file: 
    - The function tensorflow.io.read_file takes the file name as its required argument and returns the contents of the file as a tensor with type tensorflow.string. When the input file is an image, the output of tensorflow.io.read_file will be the raw byte data of the image file. Although the raw byte output represents the image's pixel data, it cannot be used directly.
    - The function tf.image.decode_image detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the appropriate operation to convert the input bytes string into a Tensor of type dtype (Pixel data)
- Grayscale:
    - The functions tfio.experimental.color.rgb_to_grayscale Converts RGB to Grayscale. An RGB image can be converted to Grayscale to reduce the channel from 3 to 1
- Image Resize:
    - We also need resized the image using the function tf.image.resize(images, size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False, name=None)
    - It takes in two required arguments: the original image's decoded data and the new size of the image, which is a tuple/list of two integers representing new_height and new_width, in that order.
- Dataset: 
     - The function tensorflow.data.Dataset was used to store all the images into a dataframe of downsized images.     


#### Further Data Processing to feed through the Neural Network models:
- Scaling our data to be between 0 and 1, the pixel values were converted to float32 by dividing by 255.
- Finally each image was reshaped to be 60x60x1: this allows the neural network to know that there is only one value instead of multiple values per pixel (grayscale).  


![image](https://user-images.githubusercontent.com/22139918/141347858-ad87fc39-6483-4f30-8b46-e97f979ab767.png)

---

# <a id = 'ModelingMethodology'>2. Modeling Methodology</b></a>

## <a id = 'Overview'>2.1 Overview</b></a>

**Archetacture Overview:** Convolutional Neural Networks consist of several neurons that are fed with input vectorized and resized image data and include weights and biases.  Neurons are trainable and learnable because each of them performs a non-linear operation on the input arrays through the ReLU function [[Ref]](https://cs231n.github.io/convolutional-networks/#overview). ReLU (Rectified Linear Unit) is ideal for this deep learning purpose because Rectified Linear Unit takes less computational time to be trained and hence this will reduce the optimization time of the neural network in the gradient descent surface [[Ref]](https://www.mygreatlearning.com/blog/relu-activation-function/). The overall score of the neural network then can be expressed with a single metric: Loss function [[Ref]](https://towardsdatascience.com/understanding-different-loss-functions-for-neural-networks-dd1ed0274718).


## <a id = 'CNNs'>2.2 Convolutional Neural Networks (CNNs/ConvNets)</b></a>

**Convolutional Neural Networks (CNNs/ConvNets):**  Compared to the regular FNNs architecture, CNNs have extra layers to apply further processing to the input images through Convolution operators, Maxpooling, and Padding. The central idea of the convolution layers is to extract the important features from the image and simplify them (or downscale). The convolution layer consists of a set of filters that take the original image and convolve them into the kernel. The following figure shows the impact of applying 3x3 kernel filter (left), 2x2 Max Pooling (middle), and 3x3 Max Pooling (right) on a 2D vectorized image ([[Fig credit]](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)).

<img width="1145" alt="image" src="https://user-images.githubusercontent.com/22139918/141344082-d075edfe-4160-4806-9c66-fc2d298e00d1.png">

In the following, the general script for the build CNNs is represented which includes 2 hidden layers (with 384 and 128 neurons). In addition, one 2D convolutional layer with 6 filters (3x3 kernel) is included. MaxPooling2D and Padding operations are tested but did not show a significant impact.  


![image](https://user-images.githubusercontent.com/22139918/142955022-a6b9832e-e469-4008-9587-5cef7878b7d9.png)

![image](https://user-images.githubusercontent.com/22139918/142955397-dafe2321-5536-4eed-b9e0-d3b1c466e507.png)




---
# <a id = 'Results'>Classification Results from CNNs model</b></a>

![image](https://user-images.githubusercontent.com/22139918/142955617-2e62f414-4050-403b-b9fd-7393c9180786.png)


In order to assess the model performance with outside dataset, we generated a set of images to test the model robustness. Although, the trained CNNs model scores very well for the Kaggle datasets, it suffers from overfitting for the external datasets. The issues can be related to several factors including a strict control environment, same hand, limited background and lightning, same perspective (distance, angle). Note that the dataset used to train the CNNs model was from the same indivitual. Hence, the background, skin color, and other parameters were generally the same for both the training and tesing sets from the Kaggle. However, the skin color and background from our external testing image set were different, which could be the main source of decrease in the accuracy.


---
# <a id = 'Recommendations'>Recommendations</b></a>

1. Crowd source imaging for a more diverse dataset

2. Train the model with RGB and not on greyscale

3. Train the model on words rather than single characters 

