import numpy as np
import tensorflow as tf

from google.cloud import storage

model = None
char2idx = {' ': 0,
 '!': 1,
 '"': 2,
 '#': 3,
 '$': 4,
 '&': 5,
 "'": 6,
 '(': 7,
 ')': 8,
 '*': 9,
 ',': 10,
 '-': 11,
 '.': 12,
 '/': 13,
 '0': 14,
 '1': 15,
 '2': 16,
 '3': 17,
 '4': 18,
 '5': 19,
 '6': 20,
 '7': 21,
 '8': 22,
 '9': 23,
 ':': 24,
 ';': 25,
 '<': 26,
 '=': 27,
 '>': 28,
 '?': 29,
 'A': 30,
 'B': 31,
 'C': 32,
 'D': 33,
 'E': 34,
 'F': 35,
 'G': 36,
 'H': 37,
 'I': 38,
 'J': 39,
 'K': 40,
 'L': 41,
 'M': 42,
 'N': 43,
 'O': 44,
 'P': 45,
 'Q': 46,
 'R': 47,
 'S': 48,
 'T': 49,
 'U': 50,
 'V': 51,
 'W': 52,
 'X': 53,
 'Y': 54,
 'Z': 55,
 '[': 56,
 ']': 57,
 'a': 58,
 'b': 59,
 'c': 60,
 'd': 61,
 'e': 62,
 'f': 63,
 'g': 64,
 'h': 65,
 'i': 66,
 'j': 67,
 'k': 68,
 'l': 69,
 'm': 70,
 'n': 71,
 'o': 72,
 'p': 73,
 'q': 74,
 'r': 75,
 's': 76,
 't': 77,
 'u': 78,
 'v': 79,
 'w': 80,
 'x': 81,
 'y': 82,
 'z': 83,
 'à': 84,
 'ç': 85,
 'è': 86,
 'é': 87,
 '—': 88,
 '�': 89}
idx2char = np.array([' ', '!', '"', '#', '$', '&', "'", '(', ')', '*', ',', '-', '.',
       '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
       '<', '=', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
       'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
       'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
       'u', 'v', 'w', 'x', 'y', 'z', 'à', 'ç', 'è', 'é', '—', '�'])

def generate_text(model, num_generate, temperature, start_string):

    input_eval = [char2idx[s] for s in start_string] # string to numbers (vectorizing)
    input_eval = tf.expand_dims(input_eval, 0) # dimension expansion
    text_generated = [] # Empty string to store our results
    model.reset_states() # Clears the hidden states in the RNN

    for i in range(num_generate): # Run a loop for number of characters to generate
        predictions = model(input_eval) # prediction for single character
        predictions = tf.squeeze(predictions, 0) # remove the batch dimension

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # The predicted character as the next input to the model
        # along with the previous hidden state
        # So the model makes the next prediction based on the previous character
        input_eval = tf.expand_dims([predicted_id], 0) 
        # Also devectorize the number and add to the generated text
        text_generated.append(idx2char[predicted_id]) 

    return (start_string + ''.join(text_generated))

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

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def handler(request):
    request_json = request.get_json()
    # For more information about CORS and CORS preflight requests, see
    # https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request
    # for more information.

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    # Model load which only happens during cold starts
    global model
    if model is None:
        download_blob('derangedmurakami-bucket', 'rnn-weights.index', '/tmp/rnn-weights.index')
        download_blob('derangedmurakami-bucket', 'rnn-weights.data-00000-of-00001', '/tmp/rnn-weights.data-00000-of-00001')
        model = build_model(90, 256, 1024, batch_size=1)
        model.load_weights('/tmp/rnn-weights')
        model.build(tf.TensorShape([1, None]))

    print(request_json)
    print(request)
    if request_json and 'message' in request_json:
        predtxt = request_json['message']
    else:
        predtxt = u'Kafka'
    predictions = generate_text(
                    model, 
                    num_generate=200, 
                    temperature=0.3, 
                    start_string=predtxt)
    print(predictions)
    
    return (predictions, 200, headers)
