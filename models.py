from keras.models import Model
from keras.layers import Input, Add,Subtract, PReLU, Conv2DTranspose, \
    Concatenate, MaxPooling2D, UpSampling2D, Dropout, concatenate, GlobalAveragePooling2D,\
    Reshape, Dense, Multiply, Activation, DepthwiseConv2D
from keras.layers.convolutional import Conv2D
from keras import backend as K
import tensorflow as tf
import os

def NADNet():
    inpt = Input(shape=(None,None,1)) #if the image is 3, it is color image. If the image is 1, it is gray image.
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = PReLU(shared_axes=[1,2])(x)
    s0 = x
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = PReLU(shared_axes=[1,2])(x)
    s1 = x
    for i in range(5):
        x0 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(1,1))(s1)
        x0 = PReLU(shared_axes=[1,2])(x0)
        x1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(3,3))(x0)
        x1 = PReLU(shared_axes=[1,2])(x1)
        x2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(5,5))(x1)
        x2 = PReLU(shared_axes=[1,2])(x2)
        dw = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', depth_multiplier=1)(x2)
        dw = PReLU(shared_axes=[1,2])(dw)
        pw = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(dw)
        pw = Activation('sigmoid')(pw)
        dpw = Multiply()([x2, pw])
        s1 = Add()([dpw, s1])
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(s1)
    x = PReLU(shared_axes=[1,2])(x)
    x = Add()([x, s0])
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = PReLU(shared_axes=[1,2])(x)
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)  # gray is 1, color is 3.
    o = Subtract()([inpt, x])
    model = Model(inputs=inpt, outputs=o)
    # model.summary() # print the size of model
    # x = torch.randn(1,1,50,50)
    # flops, params = thop.profile(NADNet(),inputs=(x,))
    # print(flops/1e9, params/1e6)
    return model