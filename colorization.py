# -*- coding: utf-8 -*-
'''
References : 
    https://blog.floydhub.com/colorizing-b&w-photos-with-neural-networks/
    https://keras.io/
    https://www.tensorflow.org/programmers_guide/
    http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf
    https://dl.acm.org/citation.cfm?id=2024190
'''

"""
Created on Sat Apr 21 17:26:18 2018

@author: sriha
"""
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers.normalization import BatchNormalization
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import  UpSampling2D, Input, Reshape, concatenate, Dense
#from keras.callbacks import TensorBoard 
from keras.models import Model
from keras.layers.core import RepeatVector
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, rgb2gray, gray2rgb, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf
from keras.layers import Conv2D

X = []
for filename in os.listdir('C:/Users/sriha/data/images/Train/'):
    X.append(img_to_array(load_img('C:/Users/sriha/data/images/Train/'+filename)))
X = np.array(X, dtype=float)
Xtrain = 1.0/255*X


#Load weights
inception = InceptionResNetV2(weights='imagenet', include_top=False)
inception.graph = tf.get_default_graph()

def conv_stack(data, filters, s):
        output = Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
        output = BatchNormalization()(output)
        return output

embed_input = Input(shape=(8, 8, 1536,))

#Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = conv_stack(encoder_input, 64, 2)
encoder_output = conv_stack(encoder_output, 128, 2)
encoder_output = conv_stack(encoder_output, 256, 2)
encoder_output = conv_stack(encoder_output, 512, 1)
encoder_output = conv_stack(encoder_output, 256, 1)

#Fusion
# y_mid: (None, 256, 28, 28)
fusion_output = conv_stack(embed_input, 512, 2)
fusion_output = conv_stack(fusion_output, 512, 2)
fusion_output = conv_stack(fusion_output, 512, 2)
fusion_output = conv_stack(fusion_output, 512, 1)
fusion_output = Reshape(([512]))(fusion_output)
print(fusion_output.shape)
fusion_output = Dense(1024)(fusion_output)
fusion_output = Dense(512)(fusion_output)
fusion_output = Dense(254)(fusion_output)
fusion_output = RepeatVector(32 * 32)(fusion_output) 
fusion_output = Reshape(([32, 32, 254]))(fusion_output)
fusion_output = concatenate([fusion_output, encoder_output], axis=3) 
fusion_output = Conv2D(256, (1, 1), activation='relu')(fusion_output) 

#Decoder
decoder_output = conv_stack(fusion_output, 128, 1)
decoder_output = UpSampling2D((2, 2))(fusion_output)
decoder_output = conv_stack(decoder_output, 64, 1)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = conv_stack(decoder_output, 32, 1)
decoder_output = conv_stack(decoder_output, 16, 1)
decoder_output = Conv2D(2, (2, 2), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
        print(embed.shape)
    return embed

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

#Generate training data
batch_size = 10

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)


#Train model      
#tensorboard = TensorBoard(log_dir="/tmp/log_dir/")
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit_generator(image_a_b_gen(batch_size), epochs=100, steps_per_epoch=1)

color_me = []
for filename in os.listdir('C:/Users/sriha/data/images/Test/'):
    color_me.append(img_to_array(load_img('C:/Users/sriha/data/images/Test/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me_embed = create_inception_embedding(color_me)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
	cur = np.zeros((256, 256, 3))
	cur[:,:,0] = color_me[i][:,:,0]
	cur[:,:,1:] = output[i]
	imsave("C:/Users/sriha/data/images/result1000/img_"+str(i)+".png", lab2rgb(cur))