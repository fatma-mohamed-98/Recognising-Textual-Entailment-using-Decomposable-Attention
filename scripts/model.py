from libraries import Input, Embedding, Dropout, Dense, keras, tf, \
                        K, concatenate, Lambda, Model, Reshape,EarlyStopping,\
                        np, pickle
from config import *
import gloveMatrix_init

def get_model(embedding_matrix, vocab_size):
  
  inp1 = Input((max_length,))
  inp2 = Input((max_length,))
  
  a = Embedding(vocab_size+1, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=max_length, trainable=False, mask_zero=True)(inp1)
  b = Embedding(vocab_size+1, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=max_length, trainable=False, mask_zero=True)(inp2)

  a = tf.math.l2_normalize(x=a, axis = 2)
  b = tf.math.l2_normalize(x=b, axis = 2)
  a = Dense(200)(a)
  a = Dropout(0.2)(a)
  b = Dense(200)(b) 
  b = Dropout(0.2)(b)                                  
  #************************************** Attend ***********************************
  aTb = keras.layers.Dot(axes=(2, 2))([a, b])
  exp_aTb = tf.math.exp(aTb)
  exp_aTb_T = tf.linalg.matrix_transpose(aTb)

  segma_row = K.sum(exp_aTb, axis=2)
  segma_row = Reshape(( max_length,-1), input_shape=segma_row.shape)(segma_row)
  beta_summation = tf.divide(exp_aTb, segma_row)
  beta = keras.layers.Dot(axes=(2, 1), name='beta')([beta_summation, b])

  segma_col = K.sum(exp_aTb, axis=1)
  segma_col = Reshape(( max_length,-1), input_shape = segma_col.shape)(segma_col)
  alpha_summation = tf.divide(exp_aTb_T, segma_col)
  alpha = keras.layers.Dot(axes=(2, 1), name='alpha')([alpha_summation, a])
    
  # ****************************** compare *******************************************
  v1_i = concatenate([a, beta])
  v2_j = concatenate([b, alpha])
  
  G1 = Dense(256, activation="relu")(v1_i) 
  G1 = Dropout(0.2)(G1)
  G1 = Dense(256, activation="relu")(G1) 
  G1 = Dropout(0.2)(G1)
  G1 = Dense(256, activation="relu", name='G1')(G1) 
  G1 = Dropout(0.2)(G1)
  
  G2 = Dense(256, activation="relu")(v2_j)
  G2 = Dropout(0.2)(G2)
  G2 = Dense(256, activation="relu")(G2)
  G2 = Dropout(0.2)(G2)
  G2 = Dense(256, activation="relu", name='G2')(G2)
  G2 = Dropout(0.2)(G2)
  #******************************** Aggregate *****************************************                       
  v1 = Lambda(lambda x: K.sum(x , axis=1))(G1)
  v2 = Lambda(lambda x: K.sum(x , axis=1))(G2)

  v = concatenate([v1, v2])
  H = Dense(128, activation="relu")(v)
  H = Dropout(0.2)(H)
  H = Dense(128, activation="relu")(H)
  H = Dropout(0.2)(H)
  H = Dense(64, activation="relu")(H)
  
  outp = Dense(3, activation="softmax", name="final_output")(H)
  
  model = Model(inputs=[inp1,inp2], outputs=outp)
  model.compile(loss='categorical_crossentropy',
                optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.05,
                                            initial_accumulator_value=0.1,
                                            epsilon=1e-07), metrics=['accuracy'])

  return model


def fitANDsave_model():

  embedding_matrix, vocab_size = gloveMatrix_init.get_gloveMatix()

  model = get_model(embedding_matrix, vocab_size)
  model.summary()

  with open(preproc_train_data_path, 'rb') as f:
        preproc_premise, preproc_hypothesis, Y_train = pickle.load(f)
        
  early = EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True)
  model_callbacks = [early]
  model_history = model.fit([preproc_premise, preproc_hypothesis],Y_train,
                                    batch_size = n_batch_size, 
                                    epochs = n_epochs,
                                    validation_split = 0.2,
                                    callbacks = model_callbacks)
  model.save(saved_model_path)
  return model

def get_testAcc(model):
  with open(preproc_test_data_path, 'rb') as f:
      preproc_premise_T, preproc_hypothesis_T, Y_test = pickle.load(f)   

  test_pred = model.predict([preproc_premise_T, preproc_hypothesis_T], batch_size=128)
  i = 0
  for x,y in zip(np.argmax(test_pred, axis=1), np.argmax(Y_test, axis=1)):
      if x == y:
          i += 1
  test_acc = i/Y_test.shape[0] * 100
  print("Accuracy on test set is: %"+str(test_acc))  