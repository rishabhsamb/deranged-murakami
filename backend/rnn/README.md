## Deranged Murakami

This was my entry project into NLP. It implements a basic character sequence RNN model trained over a corpus of 2 famous books by
Haruki Murakami (Norwegian Wood and Kafka on the Shore) to generate novel word sequences given some input sequence.

Since it only uses a rather naive character->integer encoding, the paragraphs produced make no sense and often create gibberish words as well. Instead of scrapping this, I found it kind of funny, so I left it up. 

The model was implemented in Tensorflow, which was a mistake for deployment since trying to configure Tensorflow Serving with a custom loss function is a headache for me personally to work with. I plan on reimplementing a better version of this project in the near future with PyTorch and Gated LSTMs.
