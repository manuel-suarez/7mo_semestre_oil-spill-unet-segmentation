import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from unittest import TestCase

def conv_bn_relu(input, filters, batch_norm=False):
    conv = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)

    return conv

def convolutional_block(input, filters, batch_norm=False, dropout_rate=0.0):
    # First convolutional block
    conv = conv_bn_relu(input, filters, batch_norm)
    # Second convolutional block
    conv = conv_bn_relu(conv, filters, batch_norm)
    if dropout_rate > 0:
        conv = layers.Dropout(dropout_rate)(conv)

    return conv

def build_unet(input_shape=(256, 256, 1), num_classes=1, activation='sigmoid', batch_norm=True, dropout_rate=0.0):
    inputs = layers.Input(input_shape)
    # Encoder Path: 256 -> 128 -> 64 -> 32 -> 16 -> 8
    # Encoder Block1 : convolutional + max pool : 256 -> 128
    dn_128 = convolutional_block(inputs, 64, batch_norm, dropout_rate)
    pl_64 = layers.MaxPooling2D(pool_size=(2,2))(dn_128)
    # Encoder Block2 : convolutional + max pool : 128 -> 64
    dn_64 = convolutional_block(pl_64, 128, batch_norm, dropout_rate)
    pl_32 = layers.MaxPooling2D(pool_size=(2,2))(dn_64)
    # Encoder Block3 : convolutional + max pool : 64 -> 32
    dn_32 = convolutional_block(pl_32, 256, batch_norm, dropout_rate)
    pl_16 = layers.MaxPooling2D(pool_size=(2,2))(dn_32)
    # Encoder Block4 : convolutional + max pool : 32 -> 16
    dn_16 = convolutional_block(pl_16, 512, batch_norm, dropout_rate)
    pl_8 = layers.MaxPooling2D(pool_size=(2,2))(dn_16)
    # Bottleneck
    dn_8 = convolutional_block(pl_8, 1024, batch_norm, dropout_rate)
    # Decoder path: 8 -> 16 -> 32 -> 64 -> 128 -> 256
    # Decoder Block4 : upsample + concat + convolutional : 8 -> 16
    up_16 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dn_8)
    ct_16 = layers.concatenate([up_16, dn_16], axis=3)
    dc_16 = convolutional_block(ct_16, 512, batch_norm, dropout_rate)
    # Decoder Block3 : upsample + concat + convolutional : 16 -> 32
    up_32 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dc_16)
    ct_32 = layers.concatenate([up_32, dn_32], axis=3)
    dc_32 = convolutional_block(ct_32, 256, batch_norm, dropout_rate)
    # Decoder Block2 : upsample + concat + convolutional : 32 -> 64
    up_64 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dc_32)
    ct_64 = layers.concatenate([up_64, dn_64], axis=3)
    dc_64 = convolutional_block(ct_64, 128, batch_norm, dropout_rate)
    # Decoder Block1 : upsample + concat + convolutional : 64 -> 128
    up_128 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(dc_64)
    ct_128 = layers.concatenate([up_128, dn_128], axis=3)
    dc_128 = convolutional_block(ct_128, 64, batch_norm, dropout_rate)
    # Classification layer
    outputs = layers.Conv2D(num_classes, kernel_size=(1,1))(dc_128)
    outputs = layers.BatchNormalization(axis=3)(outputs)
    outputs = layers.Activation(activation)(outputs)

    model = keras.Model(inputs, outputs, name='UNet')
    return model

