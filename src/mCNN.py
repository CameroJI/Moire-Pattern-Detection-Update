from keras.models import Model # type: ignore
from keras.layers import Input, Conv2D, AveragePooling2D, Dense, Dropout, Concatenate, Flatten, UpSampling2D, Multiply, Maximum, GlobalAveragePooling2D, BatchNormalization, ReLU, Add # type: ignore
from keras.applications import MobileNetV2  # type: ignore
from keras import regularizers

def createModel(height, width, depth):
    kernel_size_small = 3
    kernel_size_large = 7
    pool_size = 2
    conv_depth_1 = 32
    conv_depth_2 = 64
    drop_prob_1 = 0.25
    drop_prob_2 = 0.5
    hidden_size = 64

    inpLL = Input(shape=(height, width, depth))
    inpLH = Input(shape=(height, width, depth))
    inpHL = Input(shape=(height, width, depth))
    inpHH = Input(shape=(height, width, depth))

    # Convoluciones iniciales con filtros pequeños
    conv_1_LL = Conv2D(conv_depth_1, (kernel_size_small, kernel_size_small), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inpLL)
    conv_1_LH = Conv2D(conv_depth_1, (kernel_size_small, kernel_size_small), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inpLH)
    conv_1_HL = Conv2D(conv_depth_1, (kernel_size_small, kernel_size_small), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inpHL)
    conv_1_HH = Conv2D(conv_depth_1, (kernel_size_small, kernel_size_small), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inpHH)

    # Max Pooling
    pool_1_LL = AveragePooling2D(pool_size=(pool_size, pool_size))(conv_1_LL)
    pool_1_LH = AveragePooling2D(pool_size=(pool_size, pool_size))(conv_1_LH)
    pool_1_HL = AveragePooling2D(pool_size=(pool_size, pool_size))(conv_1_HL)
    pool_1_HH = AveragePooling2D(pool_size=(pool_size, pool_size))(conv_1_HH)

    # Fusion de características multi-escala
    avg_LH_HL_HH = Maximum()([pool_1_LH, pool_1_HL, pool_1_HH])
    inp_merged = Multiply()([pool_1_LL, avg_LH_HL_HH])

    # Bloques de atención o ajuste de detalle fino
    scale_1 = Conv2D(conv_depth_2, (kernel_size_small, kernel_size_small), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inp_merged)
    scale_2 = Conv2D(conv_depth_2, (kernel_size_large, kernel_size_large), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inp_merged)
    scale_3 = Conv2D(conv_depth_2, (kernel_size_large, kernel_size_large), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inp_merged)

    fused = Concatenate()([scale_1, scale_2, scale_3])
    S2 = AveragePooling2D(pool_size=(4, 4))(fused)

    drop_1 = Dropout(drop_prob_1)(S2)
    C5 = Conv2D(conv_depth_2, (kernel_size_small, kernel_size_small), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(drop_1)
    S3 = AveragePooling2D(pool_size=(pool_size, pool_size))(C5)
    C6 = Conv2D(conv_depth_2, (kernel_size_small, kernel_size_small), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(S3)
    S4 = AveragePooling2D(pool_size=(pool_size, pool_size))(C6)
    drop_2 = Dropout(drop_prob_2)(S4)
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(1, activation='sigmoid')(drop_3)  # Capa de salida para clasificación binaria

    return Model(inputs=[inpLL, inpLH, inpHL, inpHH], outputs=out)

def bottleneck_block(x, filters, strides=(1, 1), downsample=False):
    shortcut = x

    x = Conv2D(filters, (1, 1), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters * 4, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    if downsample or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters * 4, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x

def hrnet_block(x, filters, blocks, strides=(1, 1)):
    for _ in range(blocks):
        x = bottleneck_block(x, filters, strides)
    return x

def createMobileModel(height, width, depth):
    inputs = Input(shape=(height, width, depth))
    
    # Stem network
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Stage 1
    x = hrnet_block(x, 64, 4)
    
    # Stage 2
    x1 = hrnet_block(x, 32, 4)
    x2 = hrnet_block(x, 64, 4)
    x2 = UpSampling2D(size=(2, 2))(x2)
    
    x = Concatenate()([x1, x2])
    
    # Stage 3
    x1 = hrnet_block(x, 32, 4)
    x2 = hrnet_block(x, 64, 4)
    x3 = hrnet_block(x, 128, 4)
    x2 = UpSampling2D(size=(2, 2))(x2)
    x3 = UpSampling2D(size=(4, 4))(x3)
    
    x = Concatenate()([x1, x2, x3])
    
    # Stage 4
    x1 = hrnet_block(x, 32, 4)
    x2 = hrnet_block(x, 64, 4)
    x3 = hrnet_block(x, 128, 4)
    x4 = hrnet_block(x, 256, 4)
    x2 = UpSampling2D(size=(2, 2))(x2)
    x3 = UpSampling2D(size=(4, 4))(x3)
    x4 = UpSampling2D(size=(8, 8))(x4)
    
    x = Concatenate()([x1, x2, x3, x4])
    
    # Global Average Pooling and Dense Layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, x)
    
    return model