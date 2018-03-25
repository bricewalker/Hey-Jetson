from flask import Flask, jsonify, request, render_template, session, redirect, url_for, send_from_directory
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, DecimalField, IntegerField
from wtforms.validators import DataRequired, Required

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
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from scipy import signal

# Model metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Visualization
import IPython.display as ipd
#import librosa.display
from IPython.display import Markdown, display, Audio
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from flask import Flask, render_template, send_file, make_response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64

color = sns.color_palette()
sns.set_style('darkgrid')

# Flask app imports
from app import app
import make_predictions

# Setting random seeds
np.random.seed(95)
RNG_SEED = 95

class InferenceForm(FlaskForm):
    """Flask wtf Form to collect user input data"""
    partition = SelectField('Select data partition:', choices=[('train', 'train'), ('validation', 'validation'), ('test', 'test')], coerce=str)
    instance_number = IntegerField('Enter individual instance number:', validators=[Required()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST', 'PUT'])
@app.route('/index', methods=['GET', 'POST', 'PUT'])
@app.route('/index.html', methods=['GET', 'POST', 'PUT'])
def index():
    # Initializing form for user input
    form = InferenceForm()
    # Initializing variables passed to HTML files
    truth_transcription = None
    prediction_transcription = None
    raw_plot = None
    raw_shape = None
    spectrogram_plot = None
    spectrogram_shape = None
    error_rate = None
    cv_similarity = None
    tfidf_similarity = None
    # Defining functions for descriptive stats for the inference engine
    def plot_raw_audio(vis_raw_audio):
        # Plot the raw audio signal
        fig = plt.figure(figsize=(7,3))
        ax = fig.add_subplot(111)
        steps = len(vis_raw_audio)
        ax.plot(np.linspace(1, steps, steps), vis_raw_audio)
        plt.title('Raw Audio Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        figfile1 = BytesIO()
        plt.savefig(figfile1, format='png')
        figfile1.seek(0)
        raw_plot = base64.b64encode(figfile1.getvalue())
        return raw_plot.decode('utf8')

    def plot_spectrogram_feature(vis_spectrogram_feature):
        # Plot a normalized spectrogram
        fig = plt.figure(figsize=(7,3))
        ax = fig.add_subplot(111)
        im = ax.imshow(vis_spectrogram_feature, cmap=plt.cm.jet, aspect='auto')
        plt.title('Spectrogram')
        plt.ylabel('Time')
        plt.xlabel('Frequency')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        figfile2 = BytesIO()
        plt.savefig(figfile2, format='png')
        figfile2.seek(0)
        spectrogram_plot = base64.b64encode(figfile2.getvalue())
        return spectrogram_plot.decode('utf8')

    def wer_calc(ref, pred):
        # Calcualte word error rate
        d = np.zeros((len(ref) + 1) * (len(pred) + 1), dtype=np.uint16)
        d = d.reshape((len(ref) + 1, len(pred) + 1))
        for i in range(len(ref) + 1):
            for j in range(len(pred) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i
        for i in range(1, len(ref) + 1):
            for j in range(1, len(pred) + 1):
                if ref[i - 1] == pred[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        result = float(d[len(ref)][len(pred)]) / len(ref) * 100
        return result
    # Form for inference engine
    if form.validate_on_submit():
        partition = form.partition.data
        instance_number = form.instance_number.data
        # Get ground truth and predicted transcriptions
        truth_transcription = make_predictions.get_ground_truth(index=instance_number, partition=partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
        prediction_transcription = make_predictions.get_prediction(index=instance_number, partition=partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
        # Get features for visualizations
        vis_text, vis_raw_audio, vis_spectrogram_feature, vis_audio_path = make_predictions.vis_audio_features(index=instance_number, partition=partition)
        # Plot the audio waveform
        raw_plot = plot_raw_audio(vis_raw_audio)
        raw_shape = 'The shape of the waveform of the chosen audio file: ' + str(vis_raw_audio.shape)
        # Plot the spectrogram of the audio file
        spectrogram_plot = plot_spectrogram_feature(vis_spectrogram_feature)
        spectrogram_shape = 'The shape of the spectrogram of the chosen audio file: ' + str(vis_spectrogram_feature.shape)
        # Calculate cosine similarity of individual transcriptions using Count Vectorizer
        cv = CountVectorizer()
        cv_ground_truth_vec = cv.fit_transform([truth_transcription])
        cv_pred_transcription_vec = cv.transform([prediction_transcription])
        cv_similarity = cosine_similarity(cv_ground_truth_vec, cv_pred_transcription_vec)
        # Calculate cosine similarity of individual transcriptions using Tfidf Vectorizer
        tfidf = TfidfVectorizer()
        tfidf_ground_truth_vec = tfidf.fit_transform([truth_transcription])
        tfidf_pred_transcription_vec = tfidf.transform([prediction_transcription])
        tfidf_similarity = cosine_similarity(tfidf_ground_truth_vec, tfidf_pred_transcription_vec)
        # calculate word error rate of individual transcription
        error_rate = wer_calc(truth_transcription, prediction_transcription)
    # Render the html page with 
    return render_template('index.html', title='Hey, Jetson!', form=form, truth_transcription=truth_transcription, prediction_transcription=prediction_transcription, raw_plot=raw_plot, raw_shape=raw_shape,
    spectrogram_plot=spectrogram_plot, spectrogram_shape=spectrogram_shape, error_rate=error_rate, cv_similarity=cv_similarity, tfidf_similarity=tfidf_similarity)
    
@app.route('/about')
@app.route('/about.html')
def about():
    return render_template('about.html', title='Hey, Jetson!')

@app.route('/contact')
@app.route('/contact.html')
def contact():
    return render_template('contact.html', title='Hey, Jetson!')

app.secret_key = 'super_secret_key_shhhhhh'
if __name__ == '__main__':
    app.run(debug=True)

#venv\Scripts\activate
#flask run --host 0.0.0.0
