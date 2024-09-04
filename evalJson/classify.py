from PIL import Image, ImageTk
import tkinter as tk
import os
import sys
import argparse

class ImageClassifier:
    def __init__(self, txt_path):
        self.txt_path = txt_path
        # Inicializar contadores
        self.real_count_nuevo = 0
        self.ataque_count_nuevo = 0
        self.real_count_viejo = 0
        self.ataque_count_viejo = 0
        self.total_nuevo = 0
        self.total_viejo = 0
        self.real_total = 0
        self.ataque_total = 0
        
        # Cargar rutas de imágenes y etiquetas
        self.image_paths, self.labels = self.load_image_paths_and_labels()
        self.current_index = 0
        
        self.root = tk.Tk()
        self.root.title("Clasificador de Imágenes")
        
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()
        
        self.label = tk.Label(self.root, text="Clasifica la imagen")
        self.label.pack()
        
        # Configurar eventos de teclado
        self.root.bind('r', self.classify_real)
        self.root.bind('a', self.classify_ataque)
        self.root.bind('m', self.classify_mala_foto)
        
        self.show_image()
        
        self.root.mainloop()

    def load_image_paths_and_labels(self):
        image_paths = []
        labels = []
        with open(self.txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                path_label = line.split(':')
                path = path_label[0]
                path = path.split('-')[0]
                label_info = [path_label[1].strip().split('-')[0], path_label[2].split('\n')[0]]
                print(label_info)
                labels.append({
                    'Nuevo': label_info[0],
                    'Viejo': label_info[1]
                })
                image_paths.append(path)
                
                # Contar las etiquetas
                if label_info[0] == 'Real' or label_info[0] == 'Ataque':
                    self.total_nuevo += 1
                if label_info[1] == 'Real' or label_info[1] == 'Ataque':
                    self.total_viejo += 1
        
        return image_paths, labels

    def show_image(self):
        if self.current_index < len(self.image_paths):
            image_path = self.image_paths[self.current_index]
            if not os.path.isfile(image_path):
                print(f"Archivo no encontrado: {image_path}")
                self.current_index += 1
                self.show_image()
                return
            
            image = Image.open(image_path)
            image.thumbnail((800, 600))  # Redimensionar para ajustar al canvas
            self.photo = ImageTk.PhotoImage(image)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.root.update()
        else:
            self.root.quit()
            self.append_classification_results()
    
    def classify_real(self, event=None):
        self.save_classification("Real")
        self.real_total += 1
        if self.labels[self.current_index]['Nuevo'] == 'Real':
            self.real_count_nuevo += 1
        if self.labels[self.current_index]['Viejo'] == 'Real':
            self.real_count_viejo += 1
        self.next_image()
    
    def classify_ataque(self, event=None):
        self.save_classification("Ataque")
        self.ataque_total += 1
        if self.labels[self.current_index]['Nuevo'] == 'Ataque':
            self.ataque_count_nuevo += 1
        if self.labels[self.current_index]['Viejo'] == 'Ataque':
            self.ataque_count_viejo += 1
        self.next_image()
    
    def classify_mala_foto(self, event=None):
        self.save_classification("Mala Foto")
        self.next_image()

    def save_classification(self, classification):
        with open(self.txt_path, 'r') as file:
            lines = file.readlines()
        
        # Actualizar la línea correspondiente
        with open(self.txt_path, 'w') as file:
            for i, line in enumerate(lines):
                if i == self.current_index:
                    line = f"{line.strip()}\t{classification}\n"
                file.write(line)
    
    def append_classification_results(self):
        with open(self.txt_path, 'a') as file:
            file.write(f"\nContadores de aciertos:\n")
            file.write(f"Real aciertos en Nuevo: {self.real_count_nuevo} / {self.real_total}\n")
            file.write(f"Ataque aciertos en Nuevo: {self.ataque_count_nuevo} / {self.ataque_total}\n")
            file.write(f"Real aciertos en Viejo: {self.real_count_viejo} / {self.real_total}\n")
            file.write(f"Ataque aciertos en Viejo: {self.ataque_count_viejo} / {self.ataque_total}\n")
            
            if self.total_nuevo > 0:
                real_percentage_nuevo = (self.real_count_nuevo / self.real_total) * 100
                ataque_percentage_nuevo = (self.ataque_count_nuevo / self.ataque_total) * 100
                file.write(f"\nPorcentaje de aciertos en Nuevo:\n")
                file.write(f"Porcentaje de aciertos Real en Nuevo: {real_percentage_nuevo:.2f}%\n")
                file.write(f"Porcentaje de aciertos Ataque en Nuevo: {ataque_percentage_nuevo:.2f}%\n")
            
            if self.total_viejo > 0:
                real_percentage_viejo = (self.real_count_viejo / self.real_total) * 100
                ataque_percentage_viejo = (self.ataque_count_viejo / self.ataque_total) * 100
                file.write(f"\nPorcentaje de aciertos en Viejo:\n")
                file.write(f"Porcentaje de aciertos Real en Viejo: {real_percentage_viejo:.2f}%\n")
                file.write(f"Porcentaje de aciertos Ataque en Viejo: {ataque_percentage_viejo:.2f}%\n")

    def next_image(self):
        self.current_index += 1
        self.show_image()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--txtPath', type=str, help='Directory with (Moiré pattern) images.', default='./')
    
    return parser.parse_args(argv)
    
def main(args):
    txtPath = args.txtPath  
    
    ImageClassifier(txtPath) 
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))