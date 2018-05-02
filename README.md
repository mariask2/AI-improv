# AI-improv
Code to create automatically generated dialogs

You need a few libraries to run this.


Installing:
----------

Given that you have a conda environment you can install as follows:


conda install numpy

conda install scipy

conda install scikit-learn

conda install gensim

conda install -c anaconda nltk

You also need a directory called cornell_movie-dialogs_corpus that you need to position in the same directory as the code.
This corpus can be found here:
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

You also need a pre-trained word2vec space (the path to the space needs to be changed in the function 'get_space()' in 'use_gensim_space.py' )
The name of the space tested so far is 'GoogleNews-vectors-negative300.bin.gz'
and it can be found at:
https://github.com/mmihaltz/word2vec-GoogleNews-vectors

Running:
---------
First, run: python read_movie_lines.py

This will generate text files for three different minds: A-san, B-san and the audience.

Then, run: python use_gensim_space.py

This will generate the dialogs. The first time this is run, it will take a very long time, since models are generated. Next time, these models will be used, so it will be faster. (If you want to re-generate the models, remove the output from the models in the folder 'data_output', i.e., the files starting with 'nbrs_model')

