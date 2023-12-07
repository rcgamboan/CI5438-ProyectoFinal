from limpieza import *
import tensorflow as tf
import time
import sys, os
from sklearn.model_selection import train_test_split

class CheckpointManager:
   
    def __init__(self, checkpoint_dir, checkpoint_prefix, encoder, decoder, optimizer, loss_object, BATCH_SIZE, tgt_lang_tokenizer, src_lang_tokenizer):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,  
                                        encoder=encoder,
                                        decoder=decoder)
        self.manager = tf.train.CheckpointManager(self.checkpoint_prefix, self.checkpoint_dir, max_to_keep=3)
        self.loss_object = loss_object
        self.BATCH_SIZE = BATCH_SIZE
        self.tgt_lang_tokenizer = tgt_lang_tokenizer
        self.src_lang_tokenizer = src_lang_tokenizer
        self.optimizer = optimizer
        self.encoder = encoder
        self.decoder = decoder
    
    def save(self):
        self.manager.save()
    
    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
    
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))  
        loss_ = self.loss_object(real, pred)  

        mask = tf.cast(mask, dtype=loss_.dtype) 
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden) 

            dec_hidden = enc_hidden 

            dec_input = tf.expand_dims([self.tgt_lang_tokenizer.word_index['<sos>']] * self.BATCH_SIZE, 1) 

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output) 

                loss += self.loss_function(targ[:, t], predictions) 

                dec_input = tf.expand_dims(targ[:, t], 1) 

        batch_loss = (loss / int(targ.shape[1])) 

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables 

        gradients = tape.gradient(loss, variables) 
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    @tf.function
    def val_step(self, inp, targ, enc_hidden):
        loss = 0 
        enc_output, enc_hidden = self.encoder(inp, enc_hidden) 
        dec_hidden = enc_hidden
        dec_input =  tf.expand_dims([self.tgt_lang_tokenizer.word_index['<sos>']] * self.BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]): 
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output) 
            loss += self.loss_function(targ[:, t], predictions) 
            dec_input = tf.expand_dims(targ[:, t], 1) 

        batch_loss = (loss / int(targ.shape[1])) 

        return batch_loss 

    def train_and_validate(self, train_dataset, test_dataset, steps_per_epoch, val_steps_per_epoch,  EPOCHS=10):
        print("Training started...")
        for epoch in range(EPOCHS):
            start = time.time()

            #Step1: 
            enc_hidden = self.encoder.initialize_hidden_state()
            total_train_loss = 0
            total_val_loss = 0
            for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(inp, 
                                        targ, 
                                        enc_hidden)
                total_train_loss += batch_loss 

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch,
                                                                batch,
                                                                batch_loss.numpy()))
        
            for (batch, (inp, targ)) in enumerate(test_dataset.take(val_steps_per_epoch)):    
                val_batch_loss = self.val_step(inp, 
                                        targ, 
                                        enc_hidden)
                total_val_loss += val_batch_loss 

            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            
            #print('Total training loss is {:.4f}'.format(total_train_loss / steps_per_epoch))
            #print('Total validation loss is {:.4f}'.format( total_val_loss / val_steps_per_epoch))
            print(f'Time taken for epoch {epoch} : {time.time() - start} sec\n')

    
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

def main():

    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else :
        size = 5000

    src_sentences = load_clean_sentences(f'english-spanish-{size}-src.pkl')
    tgt_sentences = load_clean_sentences(f'english-spanish-{size}-tgt.pkl')

    src_sequences,src_lang_tokenizer,_ = tokenize(src_sentences)
    tgt_sequences,tgt_lang_tokenizer,_ = tokenize(tgt_sentences)

    src_vocab_size = len(src_lang_tokenizer.word_index)+1 
    tgt_vocab_size = len(tgt_lang_tokenizer.word_index)+1 

    src_train,src_test,tgt_train,tgt_test = train_test_split(src_sequences, tgt_sequences, shuffle=False, test_size=0.2)

    #Defining hyperparameters
    buffer_size=len(src_train)
    val_buffer_size = len(src_test)
    BATCH_SIZE = 64
    embedding_dim = 128
    units = 1024 
    steps_per_epoch = buffer_size//BATCH_SIZE
    val_steps_per_epoch = val_buffer_size//BATCH_SIZE

    train_dataset = tf.data.Dataset.from_tensor_slices((src_train, tgt_train))

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((src_test, tgt_test))

    test_dataset = test_dataset.batch(BATCH_SIZE)

    encoder = Encoder(src_vocab_size, embedding_dim, units, BATCH_SIZE) 

    decoder = Decoder(tgt_vocab_size, embedding_dim, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    if os.name == 'nt':
        checkpoint_dir = '../training_checkpoints'
    elif os.name == 'posix':
        checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")  
    checkpoint = CheckpointManager(checkpoint_dir, 
                                   checkpoint_prefix,
                                   encoder, 
                                   decoder, 
                                   optimizer, 
                                   loss_object, 
                                   BATCH_SIZE, 
                                   tgt_lang_tokenizer, 
                                   src_lang_tokenizer)
    
    checkpoint.train_and_validate(train_dataset, 
                                  test_dataset, 
                                  steps_per_epoch, 
                                  val_steps_per_epoch)

if __name__ == "__main__":
    main()