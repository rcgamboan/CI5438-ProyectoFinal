from limpieza import *
import tensorflow as tf
import time
import sys, os
from sklearn.model_selection import train_test_split


# Clase para representar el modelo a ser usado para la traduccion
# Maneja varios parámetros para configurar y guardar el modelo
# y parametros para el entrenamiento y la validación del modelo.
class Model:

    # Constructor de la clase Model.
    # Recibe como parámetros checkpoint_dir (directorio donde se guardará el modelo),
    # checkpoint_prefix (prefijo del nombre del archivo del modelo),
    # encoder (codificador), decoder (decodificador), optimizer (optimizador),
    # loss_object (función de pérdida), BATCH_SIZE (tamaño del lote),
    # tgt_lang_tokenizer (tokenizador del lenguaje objetivo) y
    # src_lang_tokenizer (tokenizador del lenguaje fuente).
    def __init__(self, checkpoint_dir, checkpoint_prefix, encoder, decoder, optimizer, loss_object, BATCH_SIZE, tgt_lang_tokenizer, src_lang_tokenizer):
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.loss_object = loss_object
        self.BATCH_SIZE = BATCH_SIZE
        self.tgt_lang_tokenizer = tgt_lang_tokenizer
        self.src_lang_tokenizer = src_lang_tokenizer
        self.training_time = 0
    
    # guardar el modelo
    def save(self):
        self.checkpoint.save(file_prefix = self.checkpoint_prefix)
    
    # función de pérdida personalizada que calcula la pérdida para los datos.
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))  
        loss_ = self.loss_object(real, pred)  

        mask = tf.cast(mask, dtype=loss_.dtype) 
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # decoración @tf.function para convertirla en un grafo computacional de TensorFlow, 
    # lo que acelera su ejecución. 
    # Esta función realiza el entrenamiento del modelo.
    @tf.function
    def train(self, inp, targ, enc_hidden):

        # Inicializar la pérdida en 0
        loss = 0

        # Aquí se utiliza tf.GradientTape() para rastrear las operaciones de TensorFlow
        # posteriormente se calcula el gradiente y se aplica el gradiente al optimizador.
        with tf.GradientTape() as tape:
            # Aquí se pasa la entrada al codificador para obtiene las salidas del codificador (enc_output)
            # y el estado oculto del codificador (enc_hidden).
            enc_output, enc_hidden = self.encoder(inp, enc_hidden) 

            # El estado oculto del codificador se pasa al decodificador como estado oculto inicial.
            dec_hidden = enc_hidden 

            # Aquí se inicializa la entrada del decodificador como un tensor de forma (batch_size, 1)
            dec_input = tf.expand_dims([self.tgt_lang_tokenizer.word_index['<start>']] * self.BATCH_SIZE, 1) 

            # Aquí se realiza un bucle para cada token de salida del decodificador.
            # En cada iteracion realiza una predicción utilizando el decodificador 
            # y obtiene las predicciones, el nuevo estado oculto del decodificador 
            # y las puntuaciones de atención.
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output) 

                # Calcula la perdida utilizando la función de pérdida definida.
                loss += self.loss_function(targ[:, t], predictions) 

                # Establece la entrada del decodificador para la siguiente iteracion
                #  como el token de destino actual.
                dec_input = tf.expand_dims(targ[:, t], 1) 

        # Calcula la perdida promedio para el batch actual.
        batch_loss = (loss / int(targ.shape[1])) 

        # Obtiene las variables entrenables del modelo.
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables 

        # Calcula los gradientes con respecto a las variables entrenables
        gradients = tape.gradient(loss, variables) 

        # Aplica los gradientes al optimizador para actualizar los pesos del modelo.
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    # Esta función realiza la validación del modelo.
    # Es similar a la funcion train pero no actualiza los pesos del modelo.
    @tf.function
    def validate(self, inp, targ, enc_hidden):
        loss = 0 
        enc_output, enc_hidden = self.encoder(inp, enc_hidden) 
        dec_hidden = enc_hidden
        dec_input =  tf.expand_dims([self.tgt_lang_tokenizer.word_index['<start>']] * self.BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]): 
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output) 
            loss += self.loss_function(targ[:, t], predictions) 
            dec_input = tf.expand_dims(targ[:, t], 1) 

        batch_loss = (loss / int(targ.shape[1])) 

        return batch_loss 

    # Esta función realiza el entrenamiento y la validación del modelo.
    def train_and_validate(self, train_dataset, test_dataset, steps_per_epoch, val_steps_per_epoch,  iters=10, show_loss = False):
        print(f"Entrenando modelo por {iters} iteraciones")

        # Bucle para entrenar y validar el modelo
        for iter in range(iters):
            
            # Inicializa el contador para calcular el tiempo de entrenamiento
            start = time.time()

            # Aquí se inicializa el estado oculto del codificador
            # Se utiliza el método initialize_hidden_state() definido en la clase Encoder.
            # Este método devuelve un tensor de ceros de forma (batch_size, units)
            enc_hidden = self.encoder.initialize_hidden_state()
            total_train_loss = 0
            total_validation_loss = 0

            # Bucle de entrenamiento
            for inputs, targets in train_dataset.take(steps_per_epoch):
                batch_loss = self.train(inputs, targets, enc_hidden)
                total_train_loss += batch_loss 

            # Bucle de validación
            for inputs, targets in test_dataset.take(val_steps_per_epoch):    
                val_batch_loss = self.validate(inputs, targets, enc_hidden)
                total_validation_loss += val_batch_loss 
            
            iter_time = time.time() - start
            self.training_time += iter_time
            print(f'\nTiempo en iteracion {iter} : {iter_time:.4f} sec')
            if show_loss:
                # Conjunto de datos muy pequeño para entrenamiento
                if steps_per_epoch != 0:
                    print(f'Perdida en entrenamiento: {(total_train_loss / steps_per_epoch):.4f}')
                # Conjunto de datos muy pequeño para validacion
                if val_steps_per_epoch != 0:
                    print(f'Perdida en validacion: {(total_validation_loss / val_steps_per_epoch):.4f}')
        
        print("Modelo entrenado exitosamente!")
        print("Guardando modelo")
        self.save()
        print(f"Modelo guardado exitosamente con el prefijo {sys.argv[1]}")
 

# Clase Encoder que representa la parte codificadora del modelo. 
# Define la arquitectura del codificador utilizando capas de Embedding y GRU (Gated Recurrent Unit).
# El codificador se encarga de procesar la entrada y devolver la salida y el estado oculto.
class Encoder(tf.keras.Model):

    # Constructor de la clase Encoder. 
    # Recibe como parámetros vocab_size (tamaño del vocabulario), 
    # emb_dim (dimensiones de las incrustaciones), 
    # enc_units (unidades del codificador), y batch_sz (tamaño del lote).    
    def __init__(self, vocab_size, emb_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()

        # Almacena las unidades del codificador y el tamaño del lote como atributos de la clase.
        self.enc_units = enc_units 
        self.batch_sz = batch_sz 

        # Capa de incrustación (embedding) que mapea índices de palabras a vectores densos de longitud emb_dim. 
        # La opción mask_zero=True se utiliza para enmascarar la entrada con valores cero, 
        # lo que ayuda a que el modelo ignore los tokens de relleno (<pad>) durante el entrenamiento.
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim,mask_zero=True)

        # Capa GRU (Gated Recurrent Unit) que implementa la unidad recurrente GRU.
        # Esta capa tiene enc_units unidades, devuelve secuencias (return_sequences=True) 
        # y el estado (return_state=True). Utiliza 'glorot_uniform' como inicializador de pesos.
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


# clase Decoder que representa la parte decodificadora del modelo. 
# Define la arquitectura del decodificador utilizando capas de Embedding, 
# GRU y una capa completamente conectada (Dense).
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz 
        self.dec_units = dec_units 

        # Inicializa el mecanismo de atención de BahdanauAttention, 
        # que se utilizará en el decodificador para calcular la atención entre el 
        # codificador y el decodificador.
        # Permite que el decodificador se centre en partes específicas de la secuencia de entrada
        # de esta manera, se puede mejorar la capacidad del modelo para manejar secuencias largas
        # y "aprender" del contexto de la palabra
        self.attention = BahdanauAttention(self.dec_units)
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim) 
        
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform') 
        
        # Capa densa (totalmente conectada) que realiza la proyección de las salidas de las GRU 
        # a las dimensiones del vocabulario (vocab_size), lo que permite generar las predicciones 
        # para el siguiente token en el proceso de decodificación.
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state , attention_weights

# Se define una capa de atención con el mecanismo Bahdanau, 
# la cual se utiliza en el decodificador para calcular el peso de atención 
# sobre las salidas del codificador.
# El mecanismo tiene como objetivo mejorar la capacidad del modelo para manejar 
# secuencias largas y capturar relaciones específicas entre las diferentes partes 
# de la secuencia de entrada y la de salida
# Este mecanismo utiliza una red neuronal adicional para calcular los pesos de atención. 
# Básicamente, toma el estado oculto (o contexto) de la capa recurrente actual y la salida (o contexto) 
# de la capa recurrente anterior para calcular los pesos de atención. 
# Luego, estos pesos se utilizan para ponderar la importancia de cada 
# elemento en la secuencia de entrada
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        # Capa densa (totalmente conectada) que se utiliza para calcular los pesos de atención.
        # Esta capa tiene units unidades y utiliza 'glorot_uniform' como inicializador de pesos.
        self.W1 = tf.keras.layers.Dense(units) 
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1) # fully-connected dense layer-3

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))   
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

def main():

    # python3 entrenamiento.py target_language size

    if len(sys.argv) != 3:
        print('Usage: python3 entrenamiento.py target_language size')
        sys.exit(1)
    
    # Leer los parametros de entrada
    size = int(sys.argv[2]) if len(sys.argv) > 1 else 5000

    # Cargar los datos limpios, obtenidos previamente en la limpieza
    if os.name == 'nt':
        src_sentences = load_clean_sentences(f'../data/clean_data/eng-{sys.argv[1]}-{size}-src.pkl')
        tgt_sentences = load_clean_sentences(f'../data/clean_data/eng-{sys.argv[1]}-{size}-tgt.pkl')
    elif os.name == 'posix':
        src_sentences = load_clean_sentences(f'./data/clean_data/english-{sys.argv[1]}-{size}-src.pkl')
        tgt_sentences = load_clean_sentences(f'./data/clean_data/english-{sys.argv[1]}-{size}-tgt.pkl')

    # tokenizar las oraciones
    # para poder usarlas en el model a entrenar
    # al ser tokenizadas, se convierten en secuencias de numeros
    src_sequences, src_lang_tokenizer, _ = tokenize(src_sentences)
    tgt_sequences, tgt_lang_tokenizer, _ = tokenize(tgt_sentences)

    # Se define el tamaño del vocabulario para el codificador y el decodificador.
    src_vocab_size = len(src_lang_tokenizer.word_index) + 1
    tgt_vocab_size = len(tgt_lang_tokenizer.word_index) + 1

    src_train, src_test, tgt_train, tgt_test = train_test_split(src_sequences, 
                                                                tgt_sequences, 
                                                                shuffle=False, 
                                                                test_size=0.2)

    # Parametros para el entrenamiento

    # Se define el tamaño del lote (batch_size), que representa el número de ejemplos
    # que se procesan por lote en el entrenamiento.
    BATCH_SIZE = 64

    # embedding_dim representa la dimensionalidad del espacio de incrustación 
    # para la capa de incrustación (Embedding) en el modelo. 
    # En un modelo de procesamiento de lenguaje natural, 
    # la capa de incrustación se utiliza para transformar números enteros que representan 
    # palabras (índices de vocabulario) en vectores densos de longitud embedding_dim. 
    # Estos vectores capturan relaciones semánticas y
    #  contextuales entre las palabras en un espacio vectorial de menor dimensión. 
    # A menudo, se considera como una representación de la palabra aprendida durante el entrenamiento.
    embedding_dim = 128

    # units representa la cantidad de unidades o neuronas en una capa específica 
    # del modelo, como por ejemplo, en una capa GRU (Gated Recurrent Unit) o en una capa completamente conectada. 
    # Por ejemplo, en una capa GRU, units sería el número de unidades en la capa recurrente. 
    # Estas unidades son responsables de aprender y capturar patrones, características o representaciones de mayor nivel 
    # en los datos de entrada.
    units = 1024 

    # Se define el número de pasos de entrenamiento y validación por iteracion.
    # representan el número total de lotes que se tomarán en cada iteracion durante el entrenamiento 
    # y la validación respectivamente. 
    # Estas variables se utilizan para controlar la cantidad de lotes que se procesan 
    # en cada iteracion del entrenamiento y la validación.
    train_steps_per_epoch = len(src_train)//BATCH_SIZE
    test_steps_per_epoch = len(src_test)//BATCH_SIZE


    # Se utiliza el método from_tensor_slices() para crear un Dataset
    # a partir de los datos de entrenamiento y validación.
    # Se utiliza el método shuffle() para mezclar los datos de entrenamiento.
    train_dataset = tf.data.Dataset.from_tensor_slices((src_train, tgt_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(src_train)).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((src_test, tgt_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Se inicializa el codificador y el decodificador.
    encoder = Encoder(src_vocab_size, embedding_dim, units, BATCH_SIZE) 
    decoder = Decoder(tgt_vocab_size, embedding_dim, units, BATCH_SIZE)
    
    # Se inicializa el optimizador a usar como el optimizador de Adam, 
    # que es un optimizador de descenso de gradiente estocástico
    # que se basa en la estimación adaptativa de momentos de primer y segundo orden.
    # Es eficiente en el uso de la memoria y es adecuado para problemas con muchos datos o parámetros.
    optimizer = tf.keras.optimizers.Adam()

    # SparseCategoricalCrossentropy es una función de pérdida que se utiliza para calcular la pérdida
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    if os.name == 'nt':
        checkpoint_dir = '../training_checkpoints'
    elif os.name == 'posix':
        checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, sys.argv[1])  
    checkpoint = Model(checkpoint_dir, 
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
                                  train_steps_per_epoch, 
                                  test_steps_per_epoch)

if __name__ == "__main__":
    main()