import tensorflow as tf
import numpy as np
import os
print("done importing")
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

print("preprocessing corpora")
kafka = open('../texts/kafka.txt', 'rb').read()
text1 = kafka.decode(encoding='utf-8')
text1.replace("\n", "")
norwood = open('../texts/norwood.txt', 'rb').read()
text2 = norwood.decode(encoding='utf-8')
checkpoint_dir = 'checkpoints'
text = text1+text2 # call me naive but i think this is fine !!

print("creating char dicts")
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

print("batching input sequences")
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) 
seq_length = 100 # The max. length for single input
sequences = char_dataset.batch(seq_length+1, drop_remainder=True) 

def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)

BUFFER_SIZE = 10000
BATCH_SIZE = 64

print("shuffling input sequences")
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

print("building model..")
model = build_model(
    vocab_size = len(vocab), # no. of unique characters
    embedding_dim=embedding_dim, # 256
    rnn_units=rnn_units, # 1024
    batch_size=BATCH_SIZE)  # 64 for the traning

model.summary()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

print("compiling model with custom loss...")
model.compile(optimizer='adam', loss=loss)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

print("training model")
EPOCHS = 30
history = model.fit(dataset, 
                    epochs=EPOCHS, 
                    callbacks=[checkpoint_callback])
                    
tf.train.latest_checkpoint('./checkpoints')
model = build_model(vocab_size, embedding_dim, batch_size=1, rnn_units=1024)
model.compile(optimizer='adam', loss=loss)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

MODEL_DIR = 'models'
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print('\nSaved model')