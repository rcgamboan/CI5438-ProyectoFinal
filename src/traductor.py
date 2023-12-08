from entrenamiento import Encoder, Decoder
from limpieza import load_clean_sentences, tokenize, preprocess_text
import tensorflow as tf
import os, sys
from keras.preprocessing.sequence import pad_sequences

# Clase que se encarga de traducir oraciones
class Traductor:

    # Constructor
    # Recibe los datasets con las oraciones del idioma fuente y del idioma objetivo
    # Recibe el encoder y decoder del modelo
    # Recibe los tokenizadores de cada idioma
    # Recibe la longitud maxima de las oraciones del idioma fuente y del idioma objetivo
    def __init__(self, src_sentences, tgt_sentences, encoder, decoder, tgt_lang_tokenizer, src_lang_tokenizer, max_length_src, max_length_trg):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_lang_tokenizer = tgt_lang_tokenizer
        self.src_lang_tokenizer = src_lang_tokenizer
        self.max_length_src = max_length_src
        self.max_length_trg = max_length_trg
        self.units = 1024

    # Toma una oración de entrada, la preprocesa, 
    # la convierte en un tensor usando el tokenizador del idioma fuente, 
    # y realiza la traducción utilizando el modelo entrenado.
    def translate(self,sentence):
        # Preprocesar la oracion
        sentence = preprocess_text(sentence)

        inputs = []
        line = sentence.split(' ')
        for i in line:
            # Se agregan los tokens correspondientes a cada palabra
            inputs.append(self.src_lang_tokenizer.word_index[i])
        # Se agrega padding para que todas las secuencias tengan la misma longitud
        inputs = pad_sequences([inputs],
                            maxlen=self.max_length_src,
                            padding='post')
        # Se convierte la oracion a un tensor
        inputs = tf.convert_to_tensor(inputs)

        result = ''
        # Se inicializa el hidden state del encoder
        # se utiliza para que el decoder sepa donde empezar a traducir
        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        # Se agrega el token de inicio de oracion al tensor
        dec_input = tf.expand_dims([self.tgt_lang_tokenizer.word_index['<sos>']], 0)

        for _ in range(self.max_length_trg):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                            dec_hidden,
                                                            enc_out)
            attention_weights = tf.reshape(attention_weights, (-1, ))
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.tgt_lang_tokenizer.index_word[predicted_id] + ' '

            if self.tgt_lang_tokenizer.index_word[predicted_id] == '<end>':
                return result, sentence
            dec_input = tf.expand_dims([predicted_id], 0)

        print('Input:', sentence)
        print('Predicted Translation:', result)

def main():

    # python3 traductor.py target_language size

    if len(sys.argv) != 3:
        print('Usage: python3 traductor.py target_language size')
        sys.exit(1)

    if os.name == 'nt':
        try:
            src_sentences = load_clean_sentences(f'../data/clean_data/eng-{sys.argv[1]}-{sys.argv[2]}-src.pkl')
            tgt_sentences = load_clean_sentences(f'../data/clean_data/eng-{sys.argv[1]}-{sys.argv[2]}-tgt.pkl')
        except:
            print("No existe un conjunto de datos limpio para el idioma seleccionado con la cantidad de datos especificada")
            sys.exit(1)
    elif os.name == 'posix':
        try:
            src_sentences = load_clean_sentences(f'./data/clean_data/eng-{sys.argv[1]}-{sys.argv[2]}-src.pkl')
            tgt_sentences = load_clean_sentences(f'./data/clean_data/eng-{sys.argv[1]}-{sys.argv[2]}-tgt.pkl')
        except:
            print("No existe un conjunto de datos limpio para el idioma seleccionado con la cantidad de datos especificada")
            sys.exit(1)
    

    _,src_lang_tokenizer,max_length_src = tokenize(src_sentences)
    _,tgt_lang_tokenizer,max_length_trg = tokenize(tgt_sentences)

    src_vocab_size = len(src_lang_tokenizer.word_index)+1 
    tgt_vocab_size = len(tgt_lang_tokenizer.word_index)+1 

    BATCH_SIZE = 64
    embedding_dim = 128
    units = 1024 

    if os.name == 'nt':
        checkpoint_dir = '../training_checkpoints'
    elif os.name == 'posix':
        checkpoint_dir = './training_checkpoints'
    

    optimizer = tf.keras.optimizers.Adam()
    encoder = Encoder(src_vocab_size, embedding_dim, units, BATCH_SIZE) 
    decoder = Decoder(tgt_vocab_size, embedding_dim, units, BATCH_SIZE)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

    try:
        # Se restaura el ultimo modelo entrenado
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir+sys.argv[1])).expect_partial()
    except:
        print("No existe un modelo entrenado para el idioma seleccionado con la cantidad de datos especificada")
        sys.exit(1)
    print("Modelo cargado")

    trad = Traductor(src_sentences, 
                     tgt_sentences, 
                     encoder, 
                     decoder, 
                     tgt_lang_tokenizer, 
                     src_lang_tokenizer, 
                     max_length_src, 
                     max_length_trg)
    
    while True:
        try:
            oracion = input("\nIngrese una oración o f para salir: ")
            if oracion == 'f':
                break
            trad.translate(oracion)
        except:
            print("No se puede traducir la oración")

    


if __name__ == "__main__":
    main()