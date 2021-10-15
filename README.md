# deranged-murakami
The product of learning over character sequences instead of word embeddings to try and reproduce text by Haruki Murakami

## Demo
There is a small demo site setup at http://rishabhsamb.github.io/deranged-murakami. Since currently the model is served on a Cloud function, cold starts can delay result times until ~20 seconds.
Further use should see 10x speedups as repeated invocations are warm.
