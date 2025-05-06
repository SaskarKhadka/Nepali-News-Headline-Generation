import random
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding
from app.config.tokenizer_config import tokenizer
from app.config.config import settings

parameters = {
    'VOC_SIZE': 50_000,

    'ENCODER_SEQUENCE_LENGTH': 256,
    'DECODER_SEQUENCE_LENGTH': 12,
    
    'EMBEDDING_DIMENSION': 100,
    
    'ENCODER_HIDDEN_DIM': 64,
    'DECODER_HIDDEN_DIM': 128,
    
    'DROPOUT': 0.3,

    'BATCH_SIZE': 128,
    'EPOCHS': 16,
    'EARLY_STOPPING': 3,
    'L2_REG': 0.01,
    
    'LEARNING_RATE': 1e-3,
    'GRAD_CLIP': 1.0,

    'PAD_TOKEN': '<pad>',
    'UNK_TOKEN': '<unk>',
    'SOS_TOKEN': '<s>',
    'EOS_TOKEN': '</s>',

    'PAD_TOKEN_ID': tokenizer.pad_id(),
    'UNK_TOKEN_ID': tokenizer.unk_id(),
    'SOS_TOKEN_ID': tokenizer.bos_id(),
    'EOS_TOKEN_ID': tokenizer.eos_id(),

    'ATTENTION_TYPE': 'bahdanau',

    'COVERAGE_WEIGHT': 1.0,
}

@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units

    # def build(self, input_shape):
        self.W = tf.keras.layers.Dense(self.units, use_bias=False)
        self.U = tf.keras.layers.Dense(self.units, use_bias=False)
        self.Wc = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def build(self, input_shape):
        encoder_outputs, decoder_hidden_state = input_shape
        self.W.build((decoder_hidden_state[0], 1, decoder_hidden_state[1]))
        self.U.build(encoder_outputs)
        self.Wc.build((encoder_outputs[0], encoder_outputs[1], 1, 1))
        self.V.build(encoder_outputs)

    def compute_output_shape(self, input_shape):
        encoder_outputs, decoder_hidden_state = input_shape
        return decoder_hidden_state, decoder_hidden_state, (decoder_hidden_state[0], decoder_hidden_state[1], 1, 1)
    
    def call(self, inputs):
        encoder_outputs, encoder_mask, decoder_hidden_state, coverage_vector = inputs  # Select the output for the current time step

        score = tf.reduce_sum(self.V(tf.nn.tanh(self.W(tf.expand_dims(tf.expand_dims(decoder_hidden_state, 1), 1)) + self.U(tf.expand_dims(encoder_outputs, 2)) + self.Wc(coverage_vector))), (2, 3))
        attention_weights = tf.nn.softmax(score, axis=1)

        # attention_weights = tf.squeeze(attention_weights, 2) * encoder_mask
        attention_weights = attention_weights * encoder_mask

        for_renorm = tf.reduce_sum(attention_weights, 1)

        attention_weights = attention_weights / tf.reshape(for_renorm, (-1, 1))

        coverage_vector += tf.reshape(attention_weights, [tf.shape(encoder_outputs)[0], -1, 1, 1])

        # Calculate the context vector
        context_vector = tf.reduce_sum(tf.reshape(attention_weights, [tf.shape(encoder_outputs)[0], -1, 1, 1]) * tf.expand_dims(encoder_outputs, 2), (1, 2))
        # print(context_vector, "BEF")
        # context_vector = tf.reshape(context_vector, [-1, tf.shape(encoder_outputs)[-1]])
        # print(context_vector, "AFT")
        return context_vector, attention_weights, coverage_vector
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units
        })
        return config
    
@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.encoder_embedding = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True, name='News_Embedding')
        self.encoder_lstm = Bidirectional(LSTM(self.hidden_dim, return_sequences=True, return_state=True, dropout=self.dropout_rate), name='Encoder_BiLSTM')

    def build(self, input_shape):
        self.encoder_embedding.build(input_shape)
        lstm_input = self.encoder_embedding.compute_output_shape(input_shape)
        self.encoder_lstm.build(lstm_input)

    def compute_output_shape(self, input_shape):
        lstm_input = self.encoder_embedding.compute_output_shape(input_shape)
        out, hf, cf, hb, cb = self.encoder_lstm.compute_output_shape(lstm_input)
        return out, out, (hf[0], hf[1]+hb[1]), (cf[0], cf[1]+cb[1]) 
    
    def call(self, encoder_input, training=False):
        encoder_embedding = self.encoder_embedding(encoder_input)

        encoder_mask = tf.cast(self.encoder_embedding.compute_mask(encoder_input), tf.float32)
        
        encoder_output, state_h_fwd, state_c_fwd, state_h_bwd, state_c_bwd = self.encoder_lstm(encoder_embedding, training=training)
        
        state_h = tf.concat([state_h_fwd, state_h_bwd], -1)
        state_c = tf.concat([state_c_fwd, state_c_bwd], -1)
        
        return encoder_output, encoder_mask, state_h, state_c
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config
    
@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.decoder_embedding = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True, name='Title_Embedding')
        self.decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True, dropout=self.dropout_rate, name='Decoder_LSTM')
        self.bahdanau_attention = BahdanauAttention(units=self.hidden_dim, name="Bahdanau_Attention")  
        self.decoder_dense = Dense(self.vocab_size, activation = 'softmax', name="Softmax_Layer")

    def build(self, input_shape):
        decoder_input, encoder_outputs = input_shape
        self.decoder_embedding.build(decoder_input)
        lstm_input = self.decoder_embedding.compute_output_shape(decoder_input)
        self.decoder_lstm.build(lstm_input)
        lstm_output, h, c = self.decoder_lstm.compute_output_shape(lstm_input)
        self.bahdanau_attention.build((encoder_outputs, h))
        cxt, attn, cvg = self.bahdanau_attention.compute_output_shape((encoder_outputs, h))
        self.decoder_dense.build((cxt[0], cxt[1]+h[1]))

    def compute_output_shape(self, input_shape):
        decoder_input, encoder_outputs = input_shape
        lstm_input = self.decoder_embedding.compute_output_shape(decoder_input)
        lstm_output, h, c = self.decoder_lstm.compute_output_shape(lstm_input)
        cxt, attn, cvg = self.bahdanau_attention.compute_output_shape((encoder_outputs, h))
        dist = self.decoder_dense.compute_output_shape((cxt[0], cxt[1]+h[1]))
        return dist, h, c, attn, cvg
    
    def call(self, decoder_input, encoder_output, encoder_mask, previous_states, coverage_vector, training=False):
        
        decoder_embedding = self.decoder_embedding(decoder_input)

        decoder_output, state_h, state_c, = self.decoder_lstm(decoder_embedding, initial_state=previous_states, training=training)
        context_vector, attention_weights, coverage_vector = self.bahdanau_attention([encoder_output, encoder_mask, state_h, coverage_vector])
        final_decoder_output = self.decoder_dense(tf.concat([context_vector, state_h], -1))
        
        return final_decoder_output, state_h, state_c, attention_weights, coverage_vector

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,

        })
        return config
    

@tf.keras.utils.register_keras_serializable()
def sparse_categorical_and_coverage_loss(targets, output_dist, attn_wts, coverage, coverage_weight=1):
    scce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=None)
    step_scce_loss = scce_loss(targets, output_dist)
    step_cov_loss = tf.reduce_sum(tf.minimum(attn_wts, coverage), 1)

    return tf.expand_dims(step_scce_loss, 1), tf.expand_dims(step_cov_loss, 1)

@tf.keras.utils.register_keras_serializable()
class AttentiveSeq2Seq(tf.keras.Model):
    def __init__(self, encoder, decoder, sos_token_id, teacher_forcing_ratio=0.5, coverage_weight=1.0, **kwargs):
        super(AttentiveSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.loss_fn = loss
        self.sos_token_id = sos_token_id
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.coverage_weight = coverage_weight

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.coverage_tracker = tf.keras.metrics.Mean(name="coverage_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_coverage_tracker = tf.keras.metrics.Mean(name="coverage_loss")

    def build(self, input_shape):
        encoder_input, decoder_input = input_shape
        self.encoder.build(encoder_input)
        encoder_outputs = self.encoder.compute_output_shape(encoder_input)
        self.decoder.build((decoder_input, encoder_outputs[0]))

    def compute_output_shape(self, input_shape):
        encoder_input, decoder_input = input_shape
        encoder_outputs = self.encoder.compute_output_shape(encoder_input)
        decoder_outputs = self.decoder.compute_output_shape((decoder_input, encoder_outputs))
        return encoder_outputs, decoder_outputs
    
    def call(self):
        """
        Decoder Inputs: <sos>.....
        Target: .....<eos>
        """

        return self.encoder, self.decoder

    @tf.function
    def train_step(self, inputs):

        encoder_inputs, targets = inputs
        loss = 0.
        cov_loss = 0.
        
        with tf.GradientTape() as tape:
            batch_loss = None
            batch_cov_loss = None
            target_mask = tf.cast(tf.math.not_equal(targets, 0), tf.float32)
            
            coverage_vector_next_t = tf.zeros([tf.shape(encoder_inputs)[0], tf.shape(encoder_inputs)[1], 1, 1], dtype=tf.float32)
            decoder_input = tf.fill([tf.shape(encoder_inputs)[0],], self.sos_token_id)

            encoder_outputs, encoder_mask, hidden, cell = self.encoder(encoder_inputs, training=True)
            
            for t in range(targets.shape[1]):
                final_dist, hidden, cell, attn_weights, coverage_vector = self.decoder(tf.expand_dims(decoder_input, 1), encoder_outputs, encoder_mask, [hidden, cell], coverage_vector_next_t, training=True)
                decoder_input = targets[:, t] if random.random() < self.teacher_forcing_ratio else tf.argmax(final_dist, 1)  

                step_loss, step_cov_loss = self.loss(targets[:,t], final_dist, attn_weights, tf.squeeze(tf.squeeze(coverage_vector_next_t, 3), 2), self.coverage_weight)

                if batch_loss is None:
                    batch_loss = step_loss
                    batch_cov_loss = step_cov_loss
                else:
                    batch_loss =tf.concat([batch_loss, step_loss], 1)
                    batch_cov_loss =tf.concat([batch_cov_loss, step_cov_loss], 1)
        
                coverage_vector_next_t = coverage_vector                

            loss = tf.reduce_mean(tf.reduce_sum(batch_loss*target_mask, 1) / tf.reduce_sum(target_mask, 1))
            cov_loss = tf.reduce_mean(tf.reduce_sum(batch_cov_loss*target_mask, 1) / tf.reduce_sum(target_mask, 1))
    
            loss = loss + self.coverage_weight * cov_loss

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.loss_tracker.update_state(loss)
        self.coverage_tracker.update_state(cov_loss)

        return {'loss': self.loss_tracker.result(), 'coverage_loss': self.coverage_tracker.result()}    

    @tf.function
    def test_step(self, inputs):
        encoder_inputs, targets = inputs

        batch_loss = None
        batch_cov_loss = None
        
        target_mask = tf.cast(tf.math.not_equal(targets, 0), tf.float32)
        
        coverage_vector_next_t = tf.zeros([tf.shape(encoder_inputs)[0], tf.shape(encoder_inputs)[1], 1, 1], dtype=tf.float32)
        decoder_input = tf.fill([tf.shape(encoder_inputs)[0],], self.sos_token_id)

        encoder_outputs, encoder_mask, hidden, cell = self.encoder(encoder_inputs, training=False)
        
        for t in range(targets.shape[1]):
            final_dist, hidden, cell, attn_weights, coverage_vector = self.decoder(tf.expand_dims(decoder_input, 1), encoder_outputs, encoder_mask, [hidden, cell], coverage_vector_next_t, training=False)
            decoder_input = tf.argmax(final_dist, 1)  

            step_loss, step_cov_loss = self.loss(targets[:,t], final_dist, attn_weights, tf.squeeze(tf.squeeze(coverage_vector_next_t, 3), 2), self.coverage_weight)

            if batch_loss is None:
                batch_loss = step_loss
                batch_cov_loss = step_cov_loss
            else:
                batch_loss =tf.concat([batch_loss, step_loss], 1)
                batch_cov_loss =tf.concat([batch_cov_loss, step_cov_loss], 1)
    
            coverage_vector_next_t = coverage_vector                
                    
        loss = tf.reduce_mean(tf.reduce_sum(batch_loss*target_mask, 1) / tf.reduce_sum(target_mask, 1))
        cov_loss = tf.reduce_mean(tf.reduce_sum(batch_cov_loss*target_mask, 1) / tf.reduce_sum(target_mask, 1))

        final_loss = loss + self.coverage_weight * cov_loss

        self.val_loss_tracker.update_state(final_loss)
        self.val_coverage_tracker.update_state(cov_loss)

        return {'loss': self.val_loss_tracker.result(), 'coverage_loss': self.val_coverage_tracker.result()}    
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'encoder': self.encoder,
            'decoder': self.decoder,
            'sos_token_id': self.sos_token_id,
            'teacher_forcing_ratio': self.teacher_forcing_ratio,
            'coverage_weight': self.coverage_weight,
            'loss_tracker': self.loss_tracker,
            'coverage_tracker': self.coverage_tracker,
            'val_loss_tracker': self.val_loss_tracker,
            'val_coverage_tracker': self.val_coverage_tracker
        })
        return config
    
def generate(model, news, tokenizer, encoder_seq_len, decoder_seq_len, sos_id, eos_id, pad_id):
    news_encoded = tokenizer.encode(news)

    if len(news_encoded) >= encoder_seq_len:
        news_encoded = news_encoded[:encoder_seq_len]
    else:
        news_encoded = news_encoded + [pad_id] * (encoder_seq_len - len(news_encoded))

    output_seq = []

    # outputs, attn_weights, coverage_vectors = model(encoder_inputs=tf.convert_to_tensor([news_encoded]), max_target_len=max_target_len, sos_token=sos_token, training=False)
    
    encoder_outputs, encoder_mask, hidden, cell = model.encoder(tf.convert_to_tensor([news_encoded]), training=False)

    decoder_input = tf.fill([1,], sos_id)
    coverage_vector = tf.zeros([1, encoder_seq_len, 1, 1])
    
    for t in range(decoder_seq_len):
        final_dist, hidden, cell, _, coverage_vector = model.decoder(tf.expand_dims(decoder_input, 1), encoder_outputs, encoder_mask, [hidden, cell], coverage_vector, training=False)
        decoder_input = tf.argmax(final_dist, 1)
        if decoder_input[0] == eos_id:
            break
        output_seq += decoder_input.numpy().tolist()
    return tokenizer.decode(list(output_seq))

def get_seq2seq_model():
    encoder = Encoder(parameters['VOC_SIZE'], parameters['EMBEDDING_DIMENSION'], parameters['ENCODER_HIDDEN_DIM'], parameters['DROPOUT'])
    decoder = Decoder(parameters['VOC_SIZE'], parameters['EMBEDDING_DIMENSION'], parameters['DECODER_HIDDEN_DIM'], parameters['DROPOUT'])    
    model = AttentiveSeq2Seq(encoder, decoder,parameters['SOS_TOKEN_ID'])
    model.build(((None, parameters['ENCODER_SEQUENCE_LENGTH']), (None, parameters['DECODER_SEQUENCE_LENGTH'])))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters['LEARNING_RATE'], weight_decay=parameters['L2_REG']), loss=sparse_categorical_and_coverage_loss)
    model.load_weights(settings.SEQ2SEQ_MODEL_PATH)
    print("---- Attentive Seq2Seq Loaded ----")
    return model

model = get_seq2seq_model()