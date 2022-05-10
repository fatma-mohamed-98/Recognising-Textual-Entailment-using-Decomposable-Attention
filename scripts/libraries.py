import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import  Input, Dense, RepeatVector, Reshape, Dropout, Flatten,concatenate, Lambda, Reshape

from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.embeddings import Embedding
import pandas as pd
import numpy as np
import re 
import os
import pickle
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
