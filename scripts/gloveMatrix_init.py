from libraries import np, pickle, Tokenizer,os
from config import GLOVE_DIR, EMBEDDING_DIM, data_tokens_path


def get_gloveMatix():

    with open(data_tokens_path, 'rb') as f:
        loaded_data = pickle.load(f)

    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(loaded_data) 

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:   
            std = 1.0
            mean = 0.0
            embedding_matrix[i] = std * np.random.randn(300) + mean
    vocab_size = len(tokenizer.word_index)  
    return embedding_matrix, vocab_size