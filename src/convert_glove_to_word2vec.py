# -*- coding: utf-8 -*-
"""
Convert a GloVe text format file to word2vec format for gensim KeyedVectors.
Usage:
    python convert_glove_to_word2vec.py glove.6B.100d.txt glove.6B.100d.word2vec.txt
"""

import sys
from gensim.scripts.glove2word2vec import glove2word2vec

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_glove_to_word2vec.py <glove_input.txt> <word2vec_output.txt>")
        sys.exit(1)
    glove_input_file = sys.argv[1]
    word2vec_output_file = sys.argv[2]
    glove2word2vec(glove_input_file, word2vec_output_file)
    print(f"[ok] converted {glove_input_file} -> {word2vec_output_file}")
