# Datasets
# http://www.manythings.org/anki/
# https://tatoeba.org/es/downloads
# https://www.aprendemachinelearning.com/una-sencilla-red-neuronal-en-python-con-keras-y-tensorflow/

# Ejemplos
# https://unipython.com/proyecto-desarrollar-un-modelo-de-traduccion-automatica-neuronal/
# https://www.ibidemgroup.com/edu/traduccion-automatica-texto-python/
# https://www.statmt.org/wmt19/translation-task.html#download
# https://www.kaggle.com/code/sharanharsoor/spanish-to-english-translation
# https://medium.com/@magodiasanket/implementation-of-neural-machine-translation-using-python-82f8f3b3e4f1

import os,io,sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import unicodedata

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, emb_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz 
    self.dec_units = dec_units 
    self.attention = BahdanauAttention(self.dec_units)
    
    self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim) 
    
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform') 
    self.fc = tf.keras.layers.Dense(vocab_size)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)
    return x, state , attention_weights

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units) # fully-connected dense layer-1
    self.W2 = tf.keras.layers.Dense(units) # fully-connected dense layer-2
    self.V = tf.keras.layers.Dense(1) # fully-connected dense layer-3

  def call(self, query, values):
   
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))   
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, emb_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.enc_units = enc_units 
        self.batch_sz = batch_sz 
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim,mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform') 

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state 

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

def load_data(file_path, size=None):
    text = io.open(file_path, encoding='UTF-8').read()
    lines = text.splitlines()
    pairs = []
    for i, line in enumerate(lines):
        pairs.append(line.split('\t'))
        if size is not None and i >= size:
            break
  
    # Eliminar las demas columnas y solo dejar la primera y la segunda de cada elemento de pairs
    pairs = [pair[:2] for pair in pairs]

    # lines =  # split the text into lines separated by newline # Insert Code Here ----
    # pairs =  # split each line into source and target using tabs # Insert Code Here ----

    source = np.array([source for target, source in pairs])  # extract source text into a numpy array
    target = np.array([target for target, source in pairs])  # extract target text into a numpy array

    return source, target

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

def load_sequences(path, size=None):
    src_sentences, tgt_sentences = load_data(path)
    src_sentences = [preprocess_text(w) for w in src_sentences]
    tgt_sentences = [preprocess_text(w) for w in tgt_sentences]

    if size is not None:
        src_sentences = src_sentences[:size]
        tgt_sentences = tgt_sentences[:size]

    src_sequences,src_lang_tokenizer,max_length_src = tokenize(src_sentences)
    tgt_sequences,tgt_lang_tokenizer,max_length_trg = tokenize(tgt_sentences)

    return src_sequences, tgt_sequences, src_lang_tokenizer, tgt_lang_tokenizer, max_length_src, max_length_trg

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))  
  loss_ = loss_object(real, pred)  

  mask = tf.cast(mask, dtype=loss_.dtype) 
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden) 

    dec_hidden = enc_hidden 

    dec_input = tf.expand_dims([tgt_lang_tokenizer.word_index['<sos>']] * BATCH_SIZE, 1) 

    for t in range(1, targ.shape[1]):
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output) 

      loss += loss_function(targ[:, t], predictions) 

      dec_input = tf.expand_dims(targ[:, t], 1) 

  batch_loss = (loss / int(targ.shape[1])) 

  variables = encoder.trainable_variables + decoder.trainable_variables 

  gradients = tape.gradient(loss, variables) 
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

@tf.function
def val_step(inp, targ, enc_hidden):
    loss = 0 
    enc_output, enc_hidden = encoder(inp, enc_hidden) 
    dec_hidden = enc_hidden
    dec_input =  tf.expand_dims([tgt_lang_tokenizer.word_index['<sos>']] * BATCH_SIZE, 1)

    for t in range(1, targ.shape[1]): 
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output) 
        loss += loss_function(targ[:, t], predictions) 
        dec_input = tf.expand_dims(targ[:, t], 1) 

    batch_loss = (loss / int(targ.shape[1])) 

    return batch_loss 

def train_and_validate(train_dataset, val_dataset, EPOCHS=10):
    for epoch in range(EPOCHS):
        start = time.time()

        #Step1: 
        enc_hidden = encoder.initialize_hidden_state()
        total_train_loss = 0
        total_val_loss = 0
        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_train_loss += batch_loss 

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch,
                                                            batch,
                                                            batch_loss.numpy()))
       
        for (batch, (inp, targ)) in enumerate(val_dataset.take(val_steps_per_epoch)):    
            val_batch_loss = val_step(inp, targ, enc_hidden) 
            total_val_loss += val_batch_loss 

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        #print('Total training loss is {:.4f}'.format(total_train_loss / steps_per_epoch))
        #print('Total validation loss is {:.4f}'.format( total_val_loss / val_steps_per_epoch))
        print(f'Time taken for epoch {epoch} : {time.time() - start} sec\n')

def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')
  fontdict = {'fontsize': 14}
  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  plt.show()

def evaluate(sentence):
    attention_plot = np.zeros((max_length_trg, max_length_src))
    sentence = preprocess_text(sentence)
    print(f"sentence: {sentence}")
    inputs = []
    line = sentence.split(' ')
    for i in line:
        print(f"i: {i}")
        inputs.append(src_lang_tokenizer.word_index[i])
    inputs = pad_sequences([inputs],
                          maxlen=max_length_src,
                          padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tgt_lang_tokenizer.word_index['<sos>']], 0)

    for t in range(max_length_trg):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += tgt_lang_tokenizer.index_word[predicted_id] + ' '

        if tgt_lang_tokenizer.index_word[predicted_id] == '<eos>':
            return result, sentence, attention_plot
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input:', sentence)
  print('Predicted Translation:', result)

  attention_plot = attention_plot[:len(result.split(' ')),
                                  :len(sentence.split(' '))]
  plot_attention(attention_plot, sentence.split(' '), result.split(' '))

if len(sys.argv) > 1:
  size = int(sys.argv[1])
else :
  size = 20000

file_path = "../data/spa.txt"

src_sequences, tgt_sequences, src_lang_tokenizer, tgt_lang_tokenizer, max_length_src, max_length_trg = load_sequences(file_path,size)

src_vocab_size = len(src_lang_tokenizer.word_index)+1 
tgt_vocab_size = len(tgt_lang_tokenizer.word_index)+1 

source_sequences_train,source_sequences_val,tgt_sequences_train,tgt_sequences_val = train_test_split(src_sequences, tgt_sequences, shuffle=False, test_size=0.2)

#Defining hyperparameters
buffer_size=len(source_sequences_train)
val_buffer_size = len(source_sequences_val)
BATCH_SIZE = 64
embedding_dim = 128
units = 1024 
steps_per_epoch = buffer_size//BATCH_SIZE
val_steps_per_epoch = val_buffer_size//BATCH_SIZE

train_dataset = tf.data.Dataset.from_tensor_slices((source_sequences_train, tgt_sequences_train))

train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((source_sequences_val, tgt_sequences_val))

val_dataset = val_dataset.batch(BATCH_SIZE)

encoder = Encoder(src_vocab_size, embedding_dim, units, BATCH_SIZE) 

attention_layer = BahdanauAttention(20) 

decoder = Decoder(tgt_vocab_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam() 

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')  

checkpoint_dir = './training_checkpoints'  
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")  
checkpoint = tf.train.Checkpoint(optimizer=optimizer,  
                                 encoder=encoder,
                                 decoder=decoder)

print("Training started...")
train_and_validate(train_dataset, val_dataset)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u"hace mucho frío aquí.")
translate(u'trata de averiguarlo.')
translate(u'¿todavía están en casa?')
