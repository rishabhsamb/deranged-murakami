import tensorflow as tf
import numpy as np

kafka = open('../texts/kafka.txt', 'rb').read()
text = kafka.decode(encoding='utf-8')
checkpoint_dir = 'checkpoints'

vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

loaded = tf.keras.models.load_model('./models/1', custom_objects={'loss':loss})
loaded.summary()

def generate_text(model, num_generate, temperature, start_string):
  input_eval = [char2idx[s] for s in start_string] # string to numbers (vectorizing)
  input_eval = tf.expand_dims(input_eval, 0) # dimension expansion
  print(input_eval)
  text_generated = [] # Empty string to store  results
  model.reset_states() # Clears the hidden states in the RNN

  for i in range(num_generate): #Run a loop for number of characters to generate
    predictions = model(input_eval) # prediction for single character
    predictions = tf.squeeze(predictions, 0) # remove the batch dimension

    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id], 0) 
    text_generated.append(idx2char[predicted_id]) 

  return (start_string + ''.join(text_generated))

def gentext(string: str) -> str:
  return generate_text(loaded, 2500, 2, string)