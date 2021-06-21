# https://github.com/homaralex/mc-dropout-mnist
# Comment from francis colhet: https://github.com/keras-team/keras/issues/9412#issuecomment-366487249
# What is MC dropout? https://datascience.stackexchange.com/questions/44065/what-is-monte-carlo-dropout

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras import Input, Model
import numpy as np
from matplotlib import pyplot as plt

def CNN_model():
    ''' 
    Normal CNN model
    '''
    inp = Input(shape=(32,32,1))
    x = Conv2D(32, (4,4), activation='relu')(inp)
    x = MaxPool2D((2,2))(x)
    #x = Dropout(0.25)(x)
    
    x = Conv2D(32, (4,4), activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    #x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    #x = Dropout(0.25)(x)
    x = Dense(10, activation='softmax')(x)
    
    return Model(inp, x, name = "CNN-with-normal-dropout")
    #return Model(inp, x)

    
def CNN_model_MC_dropout():
    ''' CNN mdoel with MC dropout
    '''
    inp = Input(shape=(32,32,1))
    x = Conv2D(32, (4,4), activation='relu')(inp)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x, training=True)
    
    x = Conv2D(64, (4,4), activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x, training=True)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(10, activation='softmax')(x)
    # a tf.keras.Model() for forward passes
    
    return Model(inp, x, name = "CNN-with-MC-dropout")

class MCLayer(tf.keras.layers.Layer):
    ''' Custom Layer.
        Consists of Convolution layer, fallowed by Maxpool and Dropout layers.
        Dropout is activated during test time also.
        '''
    
    def __init__(self, filters, **kwargs):
        super(MCLayer, self).__init__()
        filters = filters
        self.conv = tf.keras.layers.Conv2D(filters, (4,4), activation='relu')
        self.maxpool = tf.keras.layers.MaxPool2D((2,2))
        self.dropout = tf.keras.layers.Dropout(0.5)
        
    def call(self, inp_tensor):
        x = self.conv(inp_tensor)
        x = self.maxpool(x)
        x = self.dropout(x, training = True) 
        return x
    
class CNN_MC_with_inference(Model):
    
    def __init__(
        self,
        n_filters,
        n_classes,
        n_forward_passes=5,
        with_inference = True,
        name='CNN-with-MC-dropout-and-Inference',
        **kwargs
        ):

        super(CNN_MC_with_inference, self).__init__(name=name, **kwargs)
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.n_forward_passes = n_forward_passes
        self.with_inference = with_inference
        self.mclayer1 = MCLayer(n_filters)
        self.mclayer2 = MCLayer(n_filters * 2)
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(n_classes, activation='softmax')
        self.dropout = Dropout(0.5)
        
    def call(self, inp_tensor):
        mc_outputs = []
        for _ in range (self.n_forward_passes):
            #x = self.inputlayer(inp_tensor)
            x = self.mclayer1(inp_tensor)
            x = self.mclayer2(x)
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dropout(x)
            x = self.dense2(x)
            
            mc_outputs.append(x)

        if self.with_inference:
            output = tf.math.reduce_mean(tf.convert_to_tensor(mc_outputs), axis=0)
        else:
            output = tf.convert_to_tensor(mc_outputs)
        return output

def show_img_from_testset(ind):
    for img, lbl in test_data:
        print(lbl[ind])
        plt.imshow(tf.reshape(img[ind], [32,32]), cmap='gray')
        plt.show()
        break
        
def compare_model_outputs(data, first_img_ind, sec_img_ind, model1, model2):
    '''
    Shows outputs of model1 and model 2
    '''
    for img, lbl in data:
        new_data = tf.matmul(tf.reshape(img[first_img_ind], [32,32]), tf.reshape(img[sec_img_ind], [32,32]))
        count = 0
        while(count < 5):
            out = model1(tf.expand_dims(tf.expand_dims(new_data, 0), 3))
            print("Output by CNN without dropout ", np.argmax(out))
            out_mc = model2(tf.expand_dims(tf.expand_dims(new_data, 0), 3))
            print("Output by CNN with MC dropout ", np.argmax(out_mc))
            count += 1
        plt.imshow(new_data, cmap='gray')
        plt.show()
        break

