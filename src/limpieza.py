import sys, os
import numpy as np
import re
import unicodedata
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pickle import dump, load

# Obtener pares de oraciones de un archivo
def obtener_pares(file_path,size):

    # Abrir el archivo y leerlo
    try:
        text = open(file_path, mode='rt', encoding='UTF-8').read()
    except:
        print(f'Error al leer el archivo {file_path}')
        sys.exit(1)

    lines = text.splitlines()
    pairs = []
    # Separar las oraciones en pares
    # se itera hasta el tamaño especificado
    for i, line in enumerate(lines):
        pairs.append(line.split('\t'))
        if size is not None and i >= size:
            break
  
    # Eliminar las demas columnas y solo dejar la primera y la segunda de cada elemento de pairs
    pairs = [pair[:2] for pair in pairs]

    return pairs

# Transformar caracteres unicode en ascii
def unicode_to_ascii(s):
    normalized = unicodedata.normalize('NFD', s)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

# Preprocesar el texto
# Se convierte a minusculas, 
# se eliminan caracteres especiales y se agrega <start> 
# y <end> al inicio y final de cada oracion
# esto con el fin de que el modelo sepa cuando empezar y terminar de traducir
def preprocess_text(text):
  text = unicode_to_ascii(text.lower().strip())
  text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
  text = re.sub(r"([?.!,¿])", r" \1 ", text)
  text = re.sub(r'[" "]+', " ", text)
  text = text.rstrip().strip()
  text = '<start> ' + text + ' <end>'

  return text

# Tokenizar las oraciones
# Se convierten las oraciones a secuencias de numeros
# Se agrega padding para que todas las secuencias tengan la misma longitud
# Ademas se crea un tokenizador para cada idioma
def tokenize(sentences): 
    lang_tokenizer = Tokenizer( filters='')
    lang_tokenizer.fit_on_texts(sentences)
    sequences = lang_tokenizer.texts_to_sequences(sentences)
    max_length = max(len(s) for s in sequences)
    sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    return sequences, lang_tokenizer, max_length

# Cargar un conjunto de datos limpio
def load_clean_sentences(filename):
    return load(open(filename,'rb'))

# Guardar una lista de oraciones limpias para archivar
def save_clean_data(sentences, filename):
    dump(sentences, open(filename,'wb'))
    print('Saved: %s'% filename)

def main():
    
    # python3 limpieza.py filename size

    if len(sys.argv) != 3:
        print('Usage: python3 limpieza.py target_language size')
        sys.exit(1)

    # cargar conjunto de datos
    if os.name == 'nt':
        filename =f'../data/{sys.argv[1]+".txt"}'
    elif os.name == 'posix':
        filename =f'./data/{sys.argv[1]+".txt"}'
    

    if len(sys.argv) > 1:
        size = int(sys.argv[2])
    else:
        size = 5000

    pares = obtener_pares(filename,size)

    # Separar las oraciones del idioma fuente y del idioma objetivo
    source = np.array([source for target, source in pares])  
    target = np.array([target for target, source in pares])

    # Preprocesar las oraciones
    src_sentences = [preprocess_text(w) for w in source]
    tgt_sentences = [preprocess_text(w) for w in target]

    # Guardar pares limpios en su archivo correspondiente
    if os.name == 'nt':
        save_clean_data(src_sentences,f'../data/clean_data/eng-{sys.argv[1]}-{size}-src.pkl')
        save_clean_data(tgt_sentences,f'../data/clean_data/eng-{sys.argv[1]}-{size}-tgt.pkl')
    elif os.name == 'posix':
        save_clean_data(src_sentences,f'./data/clean_data/eng-{sys.argv[1]}-{size}-src.pkl')
        save_clean_data(tgt_sentences,f'./data/clean_data/eng-{sys.argv[1]}-{size}-tgt.pkl')
    

    

if __name__ == "__main__":
    main()