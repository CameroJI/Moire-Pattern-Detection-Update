from os import listdir
from os.path import join, exists, isdir
from readJson import getScoresFromJSON
import sys
import argparse

def main(args):
    dirPath = args.dirPath
        
    compareJsonFolders(dirPath)

def compareJsonFolders(root):
    txtDiferente = 'diferencias.txt'
    txtReales = 'Reales.txt'
    txtAtaque = 'Ataque.txt'
    
    txtDiferente = join(root, txtDiferente)
    txtReales = join(root, txtReales)
    txtAtaque = join(root, txtAtaque)
    
    i=0
    igualesAtaques = 0
    igualesReales = 0
    countAtaqueViejo = 0
    countAtaqueNuevo = 0
    iguales = 0
    diferentes = 0
    diferentesAtaqueNuevo = 0
    diferentesAtaqueViejo = 0
    diferenteslist = []
    realesList = []
    ataquesList = []
    
    for dir in listdir(root):
        dirPath = join(root, dir)
        
        if isdir(dirPath):
            fileNames = [file for file in listdir(dirPath) if '_front_large.jpg' in file]
            
            if len(fileNames) >= 2:
                numArray = ([int(''.join([char for char in name if char.isdigit()])) for name in fileNames])
                
                first = min(numArray)
                last = max(numArray)
                
                firtJson = f'{first}_front_result.json'
                lastJson = f'{last}_front_result.json'
                                    
                firstJsonWrite = join(dirPath, f'{first}_jsonPredict.json')
                lastJsonWrite = join(dirPath, f'{last}_jsonPredict.json')
                
                if existCheck(dirPath, [firtJson, lastJson, firstJsonWrite, lastJsonWrite]):
                    firtJsonObj = getScoresFromJSON(join(dirPath, firstJsonWrite))
                    result = splitJson(firtJsonObj, firstJsonWrite)
                    firtJsonObj = getScoresFromJSON(join(dirPath, firtJson))
                    result = splitJson(firtJsonObj, firtJson)
                    
                    lastJsonObj = getScoresFromJSON(join(dirPath, lastJsonWrite))
                    resultNew = splitJson(lastJsonObj, lastJsonWrite)
                    lastJsonObj = getScoresFromJSON(join(dirPath, lastJson))
                    resultOld = splitJson(lastJsonObj, lastJson)
                    
                    if resultNew == resultOld:
                        iguales+=1
                        if resultNew == 'Real':
                            igualesReales+=1
                            realesList.append(f'{dirPath}/{last}_front_large.jpg-Nuevo:{resultNew}-Viejo:{resultOld}')
                        else:
                            igualesAtaques+=1
                            ataquesList.append(f'{dirPath}/{last}_front_large.jpg-Nuevo:{resultNew}-Viejo:{resultOld}')
                        
                    else:
                        diferentes+=1
                        if resultNew == 'Ataque':
                            diferentesAtaqueNuevo+=1
                        if resultOld == 'Ataque':
                            diferentesAtaqueViejo+=1
                        diferenteslist.append(f'{dirPath}/{last}_front_large.jpg-Nuevo:{resultNew}-Viejo:{resultOld}')
                        
                    if resultNew == 'Ataque':
                        countAtaqueNuevo+=1
                        
                    if resultOld == 'Ataque':
                        countAtaqueViejo+=1
                        
                    print('-' * 90)
                    i+=1
            
    txtWrite(txtDiferente, diferenteslist)
    txtWrite(txtReales, realesList)
    txtWrite(txtAtaque, ataquesList)
    
    print(f'Porcentaje de ataques detectados:\nNuevo: {countAtaqueNuevo}  {(countAtaqueNuevo/i)*100}%\nViejo: {countAtaqueViejo} {(countAtaqueViejo/i)*100}%\n')
    print(f'Total de imágenes: {i}\n')
    print(f'Imágenes detectadas diferente: {diferentes}')
    print(f'Nuevo: {diferentesAtaqueNuevo}\nViejo: {diferentesAtaqueViejo}\n')
    
    print(f'Imágenes detectadas iguales: {iguales}\nReales: {igualesReales}\nAtaque: {igualesAtaques}')
    
def txtWrite(txtFile, imgList):
    with open(txtFile, 'w') as archivo:
        for idx, line in enumerate(imgList):
            archivo.write(f'{line}\n') if idx + 1 < len(imgList) else archivo.write(f'{line}')
            
def splitJson(json, type):
    if '_front_result.json' in type:
        scores = ['result_status', 'result_result', 'result_score','result_attack', 'result_ocurrences', 
               'raw_display_is_attack', 'raw_display_ocurrences', 'raw_retina_is_attack', 'raw_retina_ocurrences']
     
        result = 'result_result'
         
    elif '_jsonPredict.json' in type:
        scores = ['root_rootFolder','root_dir', 'root_imageNumber', 
               'results_score', 'results_ocurrences', 'results_prediction']
        
        result = 'results_prediction'
        
    else:
        print('No se reconoce el tipo de JSON')
        return
    
    i = 0
    col_width = 22
    for score in scores:
        try:
            scoreBase = '_'.join(score.split('_')[1:])
            value = json[score].iloc[0]
            print(f'{scoreBase:<{col_width}}: {value:<{col_width}}', end='')
            
            resultVal = json[result].iloc[0]
            if resultVal == 'PASS' or resultVal == 'WARNING':
                resultVal = 'Real'
                
            elif resultVal == 'FAIL':
                resultVal = 'Ataque'
            
            i+=1
        except:
            continue
        if (i + 1) % 3 == 0:
            print()
    print('\n')
    
    return resultVal

def existCheck(root, fileList):
    all_exist = all(exists(join(root, file)) for file in fileList)
    if not all_exist:
        missing_files = ' '.join([file for file in fileList if not exists(join(root, file))])
        print(f'⚠️ Archivos faltantes en la carpeta: {missing_files} ⚠️')
    return all_exist

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--dirPath', type=str, help='Directory with (Moiré pattern) images.', default='./')
        
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))