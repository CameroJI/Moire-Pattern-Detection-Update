#The convolutional layer C1 filter three 512 x 384 input images with 32 kernels of size 7 x 7 with a stride of 1 pixel. The stride of pooling layer S1 is 2 pixels. Then, the convolved images of LH and HL are merged together by taking the maximum from both the images. In the next step, the convolved image of LL is merged with the Max result by multiplying both the results (as explained in section III-B). C2-C4 has 16 kernels of size 3 x 3 with a stride of 1 pixel. S2 pools the merged features with a stride of 4. The dropout is applied to the output of S4 which has been flattened. The fully connected layer FC1 has 32 neurons and FC2 has 1 neuron. The activation of the output layer is a softmax function.
import os

from keras.models import Model # type: ignore
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Add, Multiply, Maximum # type: ignore

from keras import regularizers

def createModel(height, width, depth, num_classes):
    kernel_size_1 = 7
    kernel_size_2 = 3
    pool_size = 2
    conv_depth_1 = 32
    conv_depth_2 = 16
    drop_prob_1 = 0.25
    drop_prob_2 = 0.5
    hidden_size = 32

    inpLL = Input(shape=(height, width, depth))
    inpLH = Input(shape=(height, width, depth))
    inpHL = Input(shape=(height, width, depth))
    inpHH = Input(shape=(height, width, depth))

    conv_1_LL = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inpLL)
    conv_1_LH = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inpLH)
    conv_1_HL = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inpHL)
    conv_1_HH = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inpHH)
    pool_1_LL = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_LL)
    pool_1_LH = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_LH)
    pool_1_HL = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HL)
    pool_1_HH = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HH)

    avg_LH_HL_HH = Maximum()([pool_1_LH, pool_1_HL, pool_1_HH])
    inp_merged = Multiply()([pool_1_LL, avg_LH_HL_HH])
    C4 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inp_merged)
    S2 = MaxPooling2D(pool_size=(4, 4))(C4)
    drop_1 = Dropout(drop_prob_1)(S2)
    C5 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(drop_1)
    S3 = MaxPooling2D(pool_size=(pool_size, pool_size))(C5)
    C6 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(S3)
    S4 = MaxPooling2D(pool_size=(pool_size, pool_size))(C6)
    drop_2 = Dropout(drop_prob_1)(S4)
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(num_classes, activation='softmax')(drop_3)

    return Model(inputs=[inpLL, inpLH, inpHL, inpHH], outputs=out)