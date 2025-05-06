import tensorflow as tf
from app.config.config import settings
from app.config.tokenizer_config import tokenizer

parameters = {
    'VOC_SIZE': 50_000,

    'ENCODER_LAYERS': 6,
    'DECODER_LAYERS': 6,

    'ENCODER_SEQUENCE_LENGTH': 256,
    'DECODER_SEQUENCE_LENGTH': 12,
    
    'EMBEDDING_DIMENSION': 256,
    'ENCODER_ATTENTION_HEADS': 8,
    'DECODER_ATTENTION_HEADS': 8,
    'ENCODER_FFN_DIM': 4 * 512,
    'DECODER_FFN_DIM': 4 * 512,
    
    'DROPOUT': 0.2,

    'BATCH_SIZE': 64,
    'EPOCHS': 20,
    'EARLY_STOPPING': 4,
    'L2_REG': 0.01,
    
    'LEARNING_RATE': 1e-4,

    'LABEL_SMOOTHING': 0.1,
    'TEACHER_FORCING_RATIO': 0.5,
    'GRAD_CLIP': 1.0,

    'PAD_TOKEN': '<pad>',
    'UNK_TOKEN': '<unk>',
    'SOS_TOKEN': '<s>',
    'EOS_TOKEN': '</s>',

    'PAD_TOKEN_ID': tokenizer.pad_id(),
    'UNK_TOKEN_ID': tokenizer.unk_id(),
    'SOS_TOKEN_ID': tokenizer.bos_id(),
    'EOS_TOKEN_ID': tokenizer.eos_id(),

    'COVERAGE_WEIGHT': 1.0,
}

@tf.keras.utils.register_keras_serializable()
class Embeddings(tf.keras.layers.Layer):
    def __init__(self, d_model: int, seq_len: int, voc_size: int, dropout_rate: float = 0.1, **kwargs):
        super(Embeddings, self).__init__(**kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        self.voc_size = voc_size
        self.dropout_rate = dropout_rate

        self.input_emb = tf.keras.layers.Embedding(self.voc_size, self.d_model, name='Sequence_Embedding')
        self.positional_emb = tf.keras.layers.Embedding(self.seq_len, self.d_model, name='Positional_Embedding')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        self.input_emb.build(input_shape)
        self.positional_emb.build(input_shape)
        output_shape = self.input_emb.compute_output_shape(input_shape)
        self.dropout.build(output_shape)

    def compute_output_shape(self, input_shape):
        return self.input_emb.compute_output_shape(input_shape)

    def call(self, inputs, training=False):
        # inputs -> (batch, seq_len)
        positions = tf.repeat(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0]], axis=0) # (batch, seq_len) 
        inp_emb = self.input_emb(inputs) # (batch, seq_len, d_model)
        pos_emb = self.positional_emb(positions) # (batch, seq_len, d_model)

        return self.dropout(inp_emb + pos_emb, training=training) # (batch, seq_len, d_model)
    
@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, h: int, dropout_rate: float = 0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = self.d_model // self.h
        self.dropout_rate = dropout_rate

        self.w_q = tf.keras.layers.Dense(self.d_model) 
        self.w_k = tf.keras.layers.Dense(self.d_model) 
        self.w_v = tf.keras.layers.Dense(self.d_model) 
        self.w_o = tf.keras.layers.Dense(self.d_model)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        q, k, v = input_shape
        self.w_q.build(q)
        self.w_k.build(k)
        self.w_v.build(v)
        self.w_o.build(q)
        self.dropout.build(q)

    def compute_output_shape(self, input_shape):
        q, k, v = input_shape
        return self.dropout.compute_output_shape(q), (q[0], self.h, q[1], k[1]) 
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_score = q @ tf.transpose(k, perm=[0,1,3,2]) / tf.sqrt(tf.cast(k.shape[-1], dtype=tf.float32))
        if mask is not None:
            # attn_score += (mask * -1e9)
            attn_score = tf.where(mask==0, -1e9, attn_score) # Set very small values where mask = 0
            
        attn_wts = tf.nn.softmax(attn_score, -1) # (batch, h, seq_len, seq_len) seq_len*seq_len because self attention
        outputs = attn_wts @ v # (batch, h, seq_len, d_k)
        return outputs, attn_wts

    def call(self, q, k, v, mask=None, training=False):
        q = self.w_q(q) # (batch, seq_len, d_model)
        k = self.w_k(k)
        v = self.w_v(v)

        # Convert (batch, seq_len, d_model) to (batch, h, seq_len, d_k)
        # Split d_model into h*d_k and then transpose the 2nd and 3rd dimension
        q = tf.transpose(tf.reshape(q, [tf.shape(q)[0], tf.shape(q)[1], self.h, self.d_k]), perm=[0,2,1,3])
        k = tf.transpose(tf.reshape(k, [tf.shape(k)[0], tf.shape(k)[1], self.h, self.d_k]), perm=[0,2,1,3])
        v = tf.transpose(tf.reshape(v, [tf.shape(v)[0], tf.shape(v)[1], self.h, self.d_k]), perm=[0,2,1,3])

        outputs, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # First Convert (batch, h, seq_len, d_k) to (batch, seq_len, d_model)
        # Reverse the above operations
        # Run through Dense to get (batch, seq_len, d_model) 
        outputs = self.w_o(tf.reshape(tf.transpose(outputs, perm=[0,2,1,3]), [tf.shape(outputs)[0], tf.shape(outputs)[2], self.d_model]))

        return self.dropout(outputs, training=training), attn_weights
    
@tf.keras.utils.register_keras_serializable()
class AddAndNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddAndNorm, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.layer_norm.build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.layer_norm.compute_output_shape(input_shape)
        
    def call(self, skip_conn, output):
        return self.layer_norm(skip_conn + output) # (batch, seq_len, d_model)
    
@tf.keras.utils.register_keras_serializable()
class PositionwiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super(PositionwiseFeedForwardNetwork, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.inner = tf.keras.layers.Dense(self.d_ff, activation='relu')
        self.outer = tf.keras.layers.Dense(self.d_model)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        self.inner.build(input_shape)
        output = self.inner.compute_output_shape(input_shape)
        self.outer.build(output)
        self.dropout.build(self.outer.compute_output_shape(output))

    def compute_output_shape(self, input_shape):
        return self.outer.compute_output_shape(self.inner.compute_output_shape(input_shape))
        
    def call(self, inputs, training=False):
        x = self.inner(inputs) # (batch, seq_len, d_ff)
        x = self.outer(x) # (batch, seq_len, d_model)
        return self.dropout(x, training=training)
    
@tf.keras.utils.register_keras_serializable()
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, h:int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        self.mhsa = MultiHeadAttention(self.d_model, self.h, self.dropout_rate)
        self.add_norm1 = AddAndNorm()
        self.pffn = PositionwiseFeedForwardNetwork(self.d_model, self.d_ff, self.dropout_rate)
        self.add_norm2 = AddAndNorm()

    def build(self, input_shape):
        self.mhsa.build([input_shape, input_shape, input_shape])
        self.add_norm1.build(input_shape)
        self.pffn.build(input_shape)
        self.add_norm2.build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.mhsa.compute_output_shape([input_shape, input_shape, input_shape])

    def call(self, inputs, mask=None, training=False):
        mhsa_outputs, attn_weights = self.mhsa(inputs, inputs, inputs, mask, training=training)
        x = self.add_norm1(inputs, mhsa_outputs)
        pffn_outputs = self.pffn(x, training=training)
        x = self.add_norm2(x, pffn_outputs)
        
        return x, attn_weights
    
@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    def __init__(self, N: int, d_model: int, seq_len: int, voc_size: int, h:int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        assert N > 0
        self.N = N
        self.d_model = d_model
        self.seq_len = seq_len
        self.voc_size = voc_size
        self.h = h
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.embedding = Embeddings(self.d_model, self.seq_len, self.voc_size, self.dropout_rate)
        self.enc_layers = [EncoderBlock(self.d_model, self.h, self.d_ff, self.dropout_rate) for _ in range(self.N)]

    def build(self, input_shape):
        self.embedding.build(input_shape)
        output = self.embedding.compute_output_shape(input_shape)
        for encoder in self.enc_layers:
            encoder.build(output)

    def compute_output_shape(self, input_shape):
        return self.enc_layers[0].compute_output_shape(self.embedding.compute_output_shape(input_shape))

    def call(self, inputs, mask=None, training=False):
        attn_weights = None
        x = self.embedding(inputs, training=training)
        for encoder in self.enc_layers:
            x, attn_weights = encoder(x, mask=mask, training=training) 
        return x, attn_weights
    
@tf.keras.utils.register_keras_serializable()
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, h:int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        self.mhsa = MultiHeadAttention(self.d_model, self.h, self.dropout_rate)
        self.add_norm1 = AddAndNorm()
        self.mhca = MultiHeadAttention(self.d_model, self.h, self.dropout_rate)
        self.add_norm2 = AddAndNorm()
        self.pffn = PositionwiseFeedForwardNetwork(self.d_model, self.d_ff, self.dropout_rate)
        self.add_norm3 = AddAndNorm()

    def build(self, input_shape):
        dec_input_shape, enc_output_shape = input_shape
        self.mhsa.build([dec_input_shape, dec_input_shape, dec_input_shape])
        self.add_norm1.build(dec_input_shape)
        self.mhca.build([dec_input_shape, enc_output_shape, enc_output_shape])
        self.add_norm2.build(dec_input_shape)
        self.pffn.build(dec_input_shape)
        self.add_norm3.build(dec_input_shape)

    def compute_output_shape(self, input_shape):
        dec_input_shape, enc_output_shape = input_shape
        return self.mhca.compute_output_shape([dec_input_shape, enc_output_shape, enc_output_shape])

    def call(self, inputs, encoder_outputs, decoder_mask=None, encoder_mask=None, training=False):
        mhsa_outputs, _ = self.mhsa(inputs, inputs, inputs, mask=decoder_mask, training=training)
        x = self.add_norm1(inputs, mhsa_outputs)
        mhca_outputs, attn_weights = self.mhca(x, encoder_outputs, encoder_outputs, mask=encoder_mask, training=training)
        x = self.add_norm2(x, mhca_outputs)
        pffn_outputs = self.pffn(x, training=training)
        x = self.add_norm3(x, pffn_outputs)

        return x, attn_weights
    
@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    def __init__(self, N: int, d_model: int, seq_len: int, voc_size: int, h:int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        assert N > 0
        self.N = N
        self.d_model = d_model
        self.seq_len = seq_len
        self.voc_size = voc_size
        self.h = h
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.embedding = Embeddings(self.d_model, self.seq_len, self.voc_size, self.dropout_rate)
        self.dec_layers = [DecoderBlock(self.d_model, self.h, self.d_ff, self.dropout_rate) for _ in range(self.N)]

    def build(self, input_shape):
        dec_input_shape, enc_output_shape = input_shape
        self.embedding.build(dec_input_shape)
        output = self.embedding.compute_output_shape(dec_input_shape)
        for decoder in self.dec_layers:
            decoder.build([output, enc_output_shape])

    def compute_output_shape(self, input_shape):
        dec_input_shape, enc_output_shape = input_shape
        return self.dec_layers[0].compute_output_shape([self.embedding.compute_output_shape(dec_input_shape), enc_output_shape])

    def call(self, inputs, encoder_outputs, decoder_mask=None, encoder_mask=None, training=False):
        attn_weights = None
        x = self.embedding(inputs, training=training)
        for decoder in self.dec_layers:
            x, attn_weights = decoder(x, encoder_outputs, decoder_mask=decoder_mask, encoder_mask=encoder_mask, training=training) 
        return x, attn_weights
    
@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.layers.Layer):
    def __init__(
        self, 
        encoder_layers: int, 
        decoder_layers: int,
        d_model: int, 
        encoder_seq_len: int, 
        decoder_seq_len: int, 
        encoder_voc_size: int, 
        decoder_voc_size: int, 
        encoder_attention_heads: int, 
        decoder_attention_heads: int, 
        encoder_ffn_dim: int, 
        decoder_ffn_dim: int, 
        dropout: float = 0.1, 
        **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)
        
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        assert self.encoder_layers > 0 and self.decoder_layers > 0, "Encoder and Decoder must have atleast 1 layer"
        
        self.d_model = d_model
        self.encoder_seq_len = encoder_seq_len
        self.decoder_seq_len = decoder_seq_len
        self.encoder_voc_size = encoder_voc_size
        self.decoder_voc_size = decoder_voc_size
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.dropout = dropout

        self.encoder = Encoder(self.encoder_layers, self.d_model, self.encoder_seq_len, self.encoder_voc_size, self.encoder_attention_heads, self.encoder_ffn_dim, self.dropout)
        self.decoder = Decoder(self.decoder_layers, self.d_model, self.decoder_seq_len, self.decoder_voc_size, self.decoder_attention_heads, self.decoder_ffn_dim, self.dropout)
        self.projection = tf.keras.layers.Dense(decoder_voc_size)

    def build(self, input_shape):
        enc_input_shape, dec_input_shape = input_shape
        self.encoder.build(enc_input_shape)
        enc_output, _ = self.encoder.compute_output_shape(enc_input_shape)
        self.decoder.build([dec_input_shape, enc_output])
        dec_output, _ = self.decoder.compute_output_shape([dec_input_shape, enc_output])
        self.projection.build(dec_output)

    def compute_output_shape(self, input_shape):
        enc_input_shape, dec_input_shape = input_shape
        enc_output, enc_attn = self.encoder.compute_output_shape(enc_input_shape)
        dec_output, dec_attn = self.decoder.compute_output_shape([dec_input_shape, enc_output])
        return self.projection.compute_output_shape(dec_output), enc_attn, dec_attn

    def call(self, encoder_inputs, decoder_inputs, encoder_mask=None, decoder_mask=None, training=False):
        enc_outputs, enc_attn_wts = self.encoder(encoder_inputs, mask=encoder_mask, training=training)
        dec_outputs, dec_attn_wts = self.decoder(decoder_inputs, enc_outputs, encoder_mask=encoder_mask, decoder_mask=decoder_mask, training=training)
        logits = self.projection(dec_outputs)

        return logits, enc_attn_wts, dec_attn_wts
    
@tf.keras.utils.register_keras_serializable()
def cross_entropy_loss(targets, output_dist, mask=None, label_smoothing=0.1):
    targets = tf.keras.utils.to_categorical(targets, num_classes=output_dist.shape[-1])
    scce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=None, label_smoothing=label_smoothing)
    step_scce_loss = scce_loss(targets, output_dist)
    if mask is not None:
        step_scce_loss = tf.reduce_mean(tf.reduce_sum(step_scce_loss*mask, 1) / tf.reduce_sum(mask, 1))
    return step_scce_loss

@tf.keras.utils.register_keras_serializable()
class TransformerTrainer(tf.keras.Model):
    def __init__(self, transformer: Transformer, label_smoothing: float = 0.1, **kwargs):
        super(TransformerTrainer, self).__init__(**kwargs)
        self.transformer = transformer
        self.label_smoothing = label_smoothing
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    def build(self, input_shape):
        enc_input, dec_input, dec_output = input_shape
        self.transformer.build([enc_input, dec_input])

    def compute_output_shape(self, input_shape):
        enc_input, dec_input, dec_output = input_shape
        return self.transformer.compute_output_shape([enc_input, dec_input])

    def compute_padding_mask(self, inp):
        mask = tf.cast(tf.math.not_equal(inp, 0), tf.float32)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        return mask
    
    def compute_padding_lookahead_mask(self, decoder_inp):
        mask = tf.cast(tf.math.equal(decoder_inp, 0), tf.float32)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        return tf.cast(tf.maximum(mask, 1 - tf.linalg.band_part(tf.ones((decoder_inp.shape[-1], decoder_inp.shape[-1])), -1, 0)) == 0, tf.float32)

    def call(self, inputs):
        encoder_inputs, decoder_inputs, targets = inputs
        encoder_mask = self.compute_padding_mask(encoder_inputs)
        decoder_mask = self.compute_padding_lookahead_mask(decoder_inputs)

        return self.transformer(encoder_inputs, decoder_inputs, encoder_mask, decoder_mask)

    @tf.function
    def train_step(self, inputs):
        encoder_inputs, decoder_inputs, targets = inputs

        loss = None
        encoder_mask = self.compute_padding_mask(encoder_inputs)
        decoder_mask = self.compute_padding_lookahead_mask(decoder_inputs)

        with tf.GradientTape() as tape:
            logits, _, _ = self.transformer(encoder_inputs, decoder_inputs, encoder_mask, decoder_mask, training=True)
            loss = self.loss(targets, logits, tf.cast(tf.math.not_equal(targets, 0), tf.float32), self.label_smoothing)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.loss_tracker.update_state(loss)

        return {'loss': self.loss_tracker.result()}

    @tf.function
    def test_step(self, inputs):
        encoder_inputs, decoder_inputs, targets = inputs
        encoder_mask = self.compute_padding_mask(encoder_inputs)
        decoder_mask = self.compute_padding_lookahead_mask(decoder_inputs)
        
        logits, _, _ = self.transformer(encoder_inputs, decoder_inputs, encoder_mask, decoder_mask, training=False)
        loss = self.loss(targets, logits, mask=tf.cast(tf.math.not_equal(targets, 0), tf.float32), label_smoothing=0.0)

        self.val_loss_tracker.update_state(loss)

        return {'loss': self.val_loss_tracker.result()}
    
def generate(model, news, tokenizer, encoder_seq_len, decoder_seq_len, sos_id, eos_id, pad_id):
    news_encoded = tokenizer.encode(news)
    
    if len(news_encoded) >= encoder_seq_len:
        news_encoded = news_encoded[:encoder_seq_len]
    else:
        news_encoded = news_encoded + [pad_id] * (encoder_seq_len - len(news_encoded))

    output_seq = []

    encoder_mask = tf.cast(tf.math.not_equal([news_encoded], 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    enc_outputs, enc_attn_wts = model.transformer.encoder(tf.convert_to_tensor([news_encoded]), mask=encoder_mask, training=False)
    
    decoder_input = tf.fill([1,1], sos_id)

    for t in range(decoder_seq_len):
        decoder_mask = tf.cast(tf.math.equal(decoder_input, 0), tf.float32)
        decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]
        decoder_mask = tf.cast(tf.maximum(decoder_mask, 1 - tf.linalg.band_part(tf.ones((decoder_input.shape[-1], decoder_input.shape[-1])), -1, 0)) == 0, tf.float32)

        dec_outputs, dec_attn_wts = model.transformer.decoder(decoder_input, enc_outputs, encoder_mask=encoder_mask, decoder_mask=decoder_mask, training=False)
        final_dist = model.transformer.projection(dec_outputs[:,-1])
        curr_output = tf.expand_dims(tf.argmax(final_dist, -1, output_type=tf.int32), 1)
        if curr_output[0] == eos_id:
            break
        decoder_input = tf.concat([decoder_input, curr_output], -1)
    return tokenizer.decode(tf.squeeze(decoder_input, 0).numpy().tolist())

def get_transformer_model():
    tformer = Transformer(parameters['ENCODER_LAYERS'], parameters['DECODER_LAYERS'], parameters['EMBEDDING_DIMENSION'], parameters['ENCODER_SEQUENCE_LENGTH'], parameters['DECODER_SEQUENCE_LENGTH'], parameters['VOC_SIZE'], parameters['VOC_SIZE'], parameters['ENCODER_ATTENTION_HEADS'], parameters['DECODER_ATTENTION_HEADS'], parameters['ENCODER_FFN_DIM'], parameters['DECODER_FFN_DIM'])
    model = TransformerTrainer(tformer, parameters['LABEL_SMOOTHING'])
    model.build(((None, parameters['ENCODER_SEQUENCE_LENGTH']), (None, parameters['DECODER_SEQUENCE_LENGTH']), (None, parameters['DECODER_SEQUENCE_LENGTH'])))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters['LEARNING_RATE'], weight_decay=parameters['L2_REG']), loss=cross_entropy_loss, run_eagerly=True)
    model.load_weights(settings.TRANSFORMER_MODEL_PATH)
    print("---- Transformer Loaded ----")
    return model

model = get_transformer_model()
