from entrenamiento import Encoder, Decoder
from limpieza import load_clean_sentences, tokenize, preprocess_text
import tensorflow as tf
import os
from keras.preprocessing.sequence import pad_sequences


class Traductor:

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


    def evaluate(self,sentence):
        sentence = preprocess_text(sentence)
        print(f"sentence: {sentence}")
        inputs = []
        line = sentence.split(' ')
        for i in line:
            print(f"i: {i}")
            inputs.append(self.src_lang_tokenizer.word_index[i])
        inputs = pad_sequences([inputs],
                            maxlen=self.max_length_src,
                            padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''
        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.tgt_lang_tokenizer.word_index['<sos>']], 0)

        for t in range(self.max_length_trg):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                            dec_hidden,
                                                            enc_out)
            attention_weights = tf.reshape(attention_weights, (-1, ))
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.tgt_lang_tokenizer.index_word[predicted_id] + ' '

            if self.tgt_lang_tokenizer.index_word[predicted_id] == '<eos>':
                return result, sentence
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence

    def translate(self,sentence):
        result, sentence = self.evaluate(sentence)

        print('Input:', sentence)
        print('Predicted Translation:', result)

def main():

    src_sentences = load_clean_sentences(f'english-spanish-5000-src.pkl')
    tgt_sentences = load_clean_sentences(f'english-spanish-5000-tgt.pkl')

    _,src_lang_tokenizer,max_length_src = tokenize(src_sentences)
    _,tgt_lang_tokenizer,max_length_trg = tokenize(tgt_sentences)

    src_vocab_size = len(src_lang_tokenizer.word_index)+1 
    tgt_vocab_size = len(tgt_lang_tokenizer.word_index)+1 

    #Defining hyperparameters
    BATCH_SIZE = 64
    embedding_dim = 128
    units = 1024 

    checkpoint_dir = './training_checkpoints'  
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")  

    optimizer = tf.keras.optimizers.Adam()
    encoder = Encoder(src_vocab_size, embedding_dim, units, BATCH_SIZE) 
    decoder = Decoder(tgt_vocab_size, embedding_dim, units, BATCH_SIZE)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    trad = Traductor(src_sentences, 
                     tgt_sentences, 
                     encoder, 
                     decoder, 
                     tgt_lang_tokenizer, 
                     src_lang_tokenizer, 
                     max_length_src, 
                     max_length_trg)
    try:
        trad.translate(u"hace mucho frío aquí.")
    except:
        print("No se puede traducir la oración")

    try:
        trad.translate(u'trata de averiguarlo.')
    except:
        print("No se puede traducir la oración")

    try:
        trad.translate(u'¿todavía están en casa?')
    except:
        print("No se puede traducir la oración")

    


if __name__ == "__main__":
    main()