''' 
1- download (Glove 300D) from here:
https://drive.google.com/file/d/1-MczqBGYsMqu4B9GKMatM9LF8jhAW6qj/view?usp=sharing

2- download snli data from here:
https://data.deepai.org/snli_1.0.zip

'''

max_length = 25
EMBEDDING_DIM = 300
n_batch_size=4
n_epochs=60

learning_rate=0.05
initial_accumulator_value=0.1
epsilon=1e-07


test_path = 'snli_1.0/snli_1.0_test.txt'
train_path = 'snli_1.0/snli_1.0_train.txt'
dev_path = 'snli_1.0/snli_1.0_dev.txt'

GLOVE_DIR = 'Glove'

saved_model_path = "models/Attention_model.h5"
preproc_train_data_path = 'preproc_train_data.pickle'
preproc_test_data_path ='preproc_test_data.pickle'
data_tokens_path = 'all_data_words.pickle'


#change the following for prediction
premise_sentence = 'these girls are having a great time looking for seashells'
hypothesis_sentence = 'the girls are outside'
hypothesis_sentence2 = 'the girls are inside'
