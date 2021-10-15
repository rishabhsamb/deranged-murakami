# deranged-murakami
The product of learning over character sequences instead of word embeddings to try and reproduce text by Haruki Murakami. This was initially an educational project on RNNs and their implementations in modern ML framworks, but I thought the outputs of the model were funny enough to host.

## Demo
There is a small demo site setup at http://rishabhsamb.github.io/deranged-murakami. Since currently the model is served on a Cloud function, cold starts can delay result times until ~20 seconds. Further use should see 10x speedups as repeated invocations are warm.

I personally recommend trying names, phrases, or keywords that you might see in either _Kafka on the Shore_ or _Norwegian Wood_. The demo only produces 200 characters on top of the provided prompt, so maybe don't try and make a novel!
