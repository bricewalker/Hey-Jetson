# Flask web app imports
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_restful import reqparse
from flask_cors import CORS

# Prediction import
import make_predictions

# Common, File Based, and Math Imports
import pandas as pd
import numpy as np
import collections
import os
from os.path import isdir, join
from pathlib import Path
from subprocess import check_output
import sys
import math
import pickle
from glob import glob
import random
from random import sample
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

# Audio processing
from scipy import signal
from scipy.fftpack import dct
import soundfile
import json
from python_speech_features import mfcc
import librosa
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from scipy import signal

# Neural Network
import keras
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras import regularizers
from keras.constraints import max_norm
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Lambda, Dense, Dropout, Flatten, Embedding, merge, Activation, GRUCell, LSTMCell,SimpleRNNCell
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, Conv1D, SimpleRNN, GRU, LSTM, CuDNNLSTM, CuDNNGRU, Conv2D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ThresholdedReLU, ELU
from keras.layers import LeakyReLU, PReLU, ThresholdedReLU, ELU
from keras.layers import BatchNormalization, TimeDistributed, Bidirectional
from keras.layers import activations, Wrapper
from keras.regularizers import l2
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.callbacks import ModelCheckpoint 
from keras.utils import np_utils
from keras import constraints, initializers, regularizers
from keras.engine.topology import Layer
import keras.losses
from keras.backend.tensorflow_backend import set_session
from keras.engine import InputSpec
import tensorflow as tf 
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return "Okay, Jetson! This website can be used to  run inference on a Keras/TensorFlow trained Recurrent Neural Network for Automatic Speech Recognition. Please go to http://127.0.0.1:5000/keras/inference? to enter values. The values you can enter are partition (to identify the set of instances, either test or validation, you would like to run inference on, index (to identify the index of the individual instance you would like to use), and model (the model you want to use). An example might be http://127.0.0.1:5000/keras/inference?partition=test&index=2012?model=model_7"

@app.route('/keras')
def keras():
    return "Okay, Jetson! This website can be used to  run inference on a Keras/TensorFlow trained Recurrent Neural Network for Automatic Speech Recognition. Please go to http://127.0.0.1:5000/keras/inference? to enter values. The values you can enter are partition (to identify the set of instances, either test or validation, you would like to run inference on, index (to identify the index of the individual instance you would like to use), and model (the model you want to use). An example might be http://127.0.0.1:5000/keras/inference?partition=test&index=2012?model=model_7"

class Inference(Resource):  
    def get(self):

        parser = reqparse.RequestParser()
        parser.add_argument('partition', type=str,
        help='The set, either test or validation, you would like to make predictions on.')
        parser.add_argument('index', type=int,
        help='The index of the individual instance you would like to run inference on.')
        parser.add_argument('model', type=str,
        help='The name of the model you want to run')

        result = get_predictions(index=int(args[index]), 
                    partition='train',
                    input_to_softmax=str(args[model]), 
                    model_path='./results/' + str(args[model]) + '.h5')  

        return jsonify(result)

api.add_resource(Inference, '/keras/inference')

if __name__ == '__main__':  
    app.run(debug=True)