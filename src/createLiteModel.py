import tensorflow
from tensorflow import keras
import sys
import argparse

def main(args):
    modelPath = (args.modelPath)
    modelLitePath = (args.modelLitePath)
    
    # Cargar el modelo entrenado
    model = keras.models.load_model(modelPath)
    
    # Convertir el modelo a TensorFlow Lite
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
    
    #converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]  # Opcional: para optimizaciones
    converter.optimizations = [tensorflow.lite.Optimize.EXPERIMENTAL_MODE_AWARE_QUANTIZATION]  # Opcional: para optimizaciones

    tflite_model = converter.convert()

    # Guardar el modelo convertido
    with open(modelLitePath, 'wb') as f:
        f.write(tflite_model)

    print(f"Modelo convertido y guardado como '{modelLitePath}' con optimización")
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modelPath', type=str, help='Saved CNN model file', default='./checkpoint/cp.h5')
    parser.add_argument('--modelLitePath', type=str, help='Path to save CNN model lite file', default='./checkpoint/cp.tflite')
    
    return parser.parse_args(argv)

    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))