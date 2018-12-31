import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
from time import time
from random import random


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class Model(tf.keras.Model):
    
    def __init__(self):
        super(Model, self).__init__()
        #64 32
        
        network_size = 24
        
        self.input_conv1 = Conv2D(filters = network_size,kernel_size=[1,1],activation=None,padding="same")
        
        self.convs1 = []
        self.convs2 = []
        self.residual_count = 5
        
        for i in range(self.residual_count):
            self.convs1.append(Conv2D(filters = network_size,kernel_size=[5,5],activation=None,padding="valid"))
            self.convs2.append(Conv2D(filters = network_size,kernel_size=[5,5],activation=None,padding="valid"))
        
        self.outconv1 = Conv2D(filters = 128,kernel_size=[3,3],activation=tf.nn.elu,padding="same")
        
        self.outconv2 = Conv2D(filters = 128,kernel_size=[1,1],activation=tf.nn.elu,padding="same")
	
        self.pool = MaxPool2D()
        self.bn = BatchNormalization()
        
        self.policy = Conv2D(filters = 6,kernel_size = [1,1],activation=tf.nn.softmax,padding="same",kernel_initializer = normalized_columns_initializer(0.001))
        self.value= Conv2D(filters = 1,kernel_size = [1,1],activation=None,padding="same",kernel_initializer = normalized_columns_initializer(0.001))
    
    def good_pad(self,layer,distance):
        size = layer.shape[1]
        layer = tf.tile(layer,[1,3,3,1])
        return layer[:,size-distance:2*size+distance,size-distance:2*size+distance,:]
    
    def clear_time(self):
        self.old_time = time()
    
    def check_time(self,s):
        new_time = time()
        print(s+":"+str(new_time-self.old_time))
        self.old_time = new_time
    
    @tf.contrib.eager.defun
    def call(self, map_data, debug = False, drop = 0):
        
        if debug:
            self.clear_time()
        
        map_data = tf.to_float(map_data)
        
        result = self.input_conv1(map_data)
        
        if debug:
            x = float(result[0,0,0,0])
            self.check_time("input")
        
        for i in range(self.residual_count):
            old_result = result
            result = self.bn(result)
            result = tf.nn.relu(result)
            result = Dropout(drop)(result)
            result = self.good_pad(result,2)
            result = self.convs1[i](result)
            result = self.bn(result)
            result = tf.nn.relu(result)
            result = Dropout(drop)(result)
            result = self.good_pad(result,2)
            result = self.convs2[i](result)
            result = result + old_result#tf.nn.elu(result+old_result)
        
        result = Dropout(drop)(result)
        
        result = self.bn(result)
        
        if debug:
            x = float(result[0,0,0,0])
            self.check_time("residuals")
        
        result = self.outconv1(result)
        result = Dropout(drop)(result)
        result = self.outconv2(result)
        result = Dropout(drop)(result)
        #result = self.pooling(result)
        policy = self.policy(result)
        
        value = self.value(result)
        
        if debug:
            x = float(value[0,0,0,0])
            x = float(policy[0,0,0,0])
            self.check_time("outputs")
        
        return policy,value
            
