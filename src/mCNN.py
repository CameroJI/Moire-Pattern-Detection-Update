from keras.models import Model # type: ignore
from keras.layers import Input, Conv2D, AveragePooling2D, Dense, Dropout, Concatenate, Flatten, Multiply, Maximum, GlobalAveragePooling2D # type: ignore
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

def createModel_mobileNetV2(height, width, depth=7):
    input_tensor = Input(shape=(height, width, depth))
    
    base_model = MobileNetV2(input_shape=(height, width, depth), include_top=False, weights=None)
    
    x = base_model(input_tensor)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=input_tensor, outputs=output_tensor)