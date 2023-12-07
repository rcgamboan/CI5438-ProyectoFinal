import sys
import numpy as np
import re
import unicodedata
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pickle import dump, load

def obtener_pares(file_path,size):

    text = open(file_path, mode='rt', encoding='UTF-8').read()
    lines = text.splitlines()
    pairs = []
    for i, line in enumerate(lines):
        pairs.append(line.split('\t'))
        if size is not None and i >= size:
            break
  
    # Eliminar las demas columnas y solo dejar la primera y la segunda de cada elemento de pairs
    pairs = [pair[:2] for pair in pairs]

    return pairs

def unicode_to_ascii(s):
    normalized = unicodedata.normalize('NFD', s)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

def preprocess_text(text):
  text = unicode_to_ascii(text.lower().strip())
  text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
  text = re.sub(r"([?.!,¿])", r" \1 ", text)
  text = re.sub(r'[" "]+', " ", text)
  text = text.rstrip().strip()
  text = '<sos> ' + text + ' <eos>'

  return text

def tokenize(sentences): 
    lang_tokenizer = Tokenizer( filters='')
    lang_tokenizer.fit_on_texts(sentences)
    sequences = lang_tokenizer.texts_to_sequences(sentences)
    max_length = max(len(s) for s in sequences)
    sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    return sequences, lang_tokenizer, max_length

# cargar un conjunto de datos limpio
def load_clean_sentences(filename):
 return load(open(filename,'rb'))

# guardar una lista de oraciones limpias para archivar
def save_clean_data(sentences, filename):
    dump(sentences, open(filename,'wb'))
    print('Saved: %s'% filename)

def main():
    # cargar conjunto de datos
    filename ='../data/spa.txt'

    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else :
        size = 5000

    pares = obtener_pares(filename,size)

    source = np.array([source for target, source in pares])  
    target = np.array([target for target, source in pares])

    src_sentences = [preprocess_text(w) for w in source]
    tgt_sentences = [preprocess_text(w) for w in target]

    # guardar pares limpios en su archivo correspondiente
    save_clean_data(src_sentences,f'english-spanish-{size}-src.pkl')
    save_clean_data(tgt_sentences,f'english-spanish-{size}-tgt.pkl')

    

if __name__ == "__main__":
    main()