from libraries import pd, re, preprocessing, pickle,Tokenizer, os, pad_sequences
from config import *

def brief_clean(txt):
    return re.sub("[^A-Za-z']+", ' ', str(txt)).lower().replace("'", '')

def get_data (path):
    data = pd.read_csv (path, delimiter = ' ')
    data = data[data['gold_label'] != '-' ]
    data = data[data['sentence2'].notna()]
    data = data[data['sentence1'].notna()]
    data['sentence1_clean'] = data.sentence1.apply(brief_clean)
    data['sentence2_clean'] = data.sentence2.apply(brief_clean)
    data = data[["gold_label", "sentence1_clean", "sentence2_clean" ]]
    return data

def get_oneHotVec(df_col, tokenizer):
    preprocess_x = tokenizer.texts_to_sequences(df_col)
    preprocess_x = pad_sequences(preprocess_x , maxlen = max_length, padding='post')
    return preprocess_x  

def get_oneHotTarget(train_labels, test_labels):
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_labels)
    Y_train = lb.transform(train_labels)
    Y_test = lb.transform(test_labels)
    return Y_train,Y_test


def save_processed_data():
    if not (os.path.exists(preproc_test_data_path) and os.path.exists(data_tokens_path) and os.path.exists(preproc_train_data_path)):
        print('waiting')
        train_df = pd.concat([get_data(train_path), get_data(dev_path)], ignore_index=True)
        test_df = get_data(test_path)
        premise = list(train_df["sentence1_clean"])
        hypothesis = list(train_df["sentence2_clean"])
        premise_test = list(test_df["sentence1_clean"])
        hypothesis_test = list(test_df["sentence2_clean"])

        all_words = list(set(premise + hypothesis + premise_test + hypothesis_test))
        with open(data_tokens_path, 'wb') as f:
            pickle.dump(all_words, f)

        tokenizer = Tokenizer(char_level=False)
        tokenizer.fit_on_texts(all_words)  

        preproc_premise = get_oneHotVec(premise,tokenizer)  
        preproc_hypothesis = get_oneHotVec(hypothesis,tokenizer)

        preproc_premise_test = get_oneHotVec(premise_test,tokenizer)  
        preproc_hypothesis_test = get_oneHotVec(hypothesis_test,tokenizer)  

        train_labels = list(train_df["gold_label"])
        test_labels = list(test_df["gold_label"])

        Y_train,Y_test = get_oneHotTarget(train_labels, test_labels)

        train_data = [preproc_premise, preproc_hypothesis, Y_train]
        test_data = [preproc_premise_test, preproc_hypothesis_test, Y_test]
        
        with open(preproc_train_data_path, 'wb') as f:
            pickle.dump(train_data, f)
    
        with open(preproc_test_data_path, 'wb') as f:
            pickle.dump(test_data, f)

        print('data saved successfully')
    else:
        print('preprocessed data aleady saved before!')    
       