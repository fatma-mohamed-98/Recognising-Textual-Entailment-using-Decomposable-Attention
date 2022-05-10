from libraries import re, pd, pad_sequences, tf, Tokenizer, pickle
from config import *
from model import *

loaded_model = tf.keras.models.load_model(saved_model_path)

def brief_clean(txt):
    return re.sub("[^A-Za-z']+", ' ', str(txt)).lower().replace("'", '')

def get_clean_input(premise_sentence, premise_sentence2,\
    hypothesis_sentence, hypothesis_sentence2):

    input_list = [[premise_sentence, hypothesis_sentence ],\
        [premise_sentence2,hypothesis_sentence2 ]]
    df = pd.DataFrame (input_list, columns = ['p', 'h'])
    df['sentence1_clean'] = df.p.apply(brief_clean)
    df['sentence2_clean'] = df.h.apply(brief_clean)
    return df

df = get_clean_input(premise_sentence, premise_sentence,\
     hypothesis_sentence, hypothesis_sentence2)
  
def get_oneHotVec(df_col, tokenizer):
    preprocess_x = tokenizer.texts_to_sequences(df_col)
    preprocess_x = pad_sequences(preprocess_x , maxlen = max_length, padding='post')
    return preprocess_x 

with open(data_tokens_path, 'rb') as f:
    loaded_data = pickle.load(f)

tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(loaded_data)

premise = list(df["sentence1_clean"])
hypothesis = list(df["sentence2_clean"])

preproc_premise = get_oneHotVec(premise,tokenizer)  
preproc_hypothesis = get_oneHotVec(hypothesis,tokenizer)

prediction = loaded_model.predict([preproc_premise, preproc_hypothesis])
labels = ['contradiction', 'entailment', 'neutral']
print('prediction for first row (outside): ',labels[prediction[0].argmax()])
print('prediction for second row (inside): ', labels[prediction[1].argmax()])
