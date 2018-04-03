from flask import Flask, jsonify, request, render_template, session, redirect, url_for, send_from_directory, Markup
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

# Model metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Visualization
import IPython.display as ipd
from IPython.display import Markdown, display, Audio
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
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

def plot_raw_audio(sample_rate, samples):
    # Plot the raw audio signal
    time = np.arange(0, float(samples.shape[0]), 1) / sample_rate
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    ax.plot(time, samples, linewidth=1, alpha=0.7, color='#76b900')
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
    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(vis_spectrogram_feature.T, cmap=plt.cm.viridis, aspect='auto', origin='lower')
    plt.title('Normalized Log Spectrogram')
    plt.ylabel('Frequency')
    plt.xlabel('Time (s)')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    figfile2 = BytesIO()
    plt.savefig(figfile2, format='png')
    figfile2.seek(0)
    spectrogram_plot = base64.b64encode(figfile2.getvalue())
    return spectrogram_plot.decode('utf8')

def log_spectrogram_feature(samples, sample_rate, window_size=20, step_size=10, eps=1e-14):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(samples,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    freqs = (freqs*2)
    return freqs, times, np.log(spec.T.astype(np.float64) + eps)

def plot_log_spectrogram_feature(freqs, times, log_spectrogram):
    fig = plt.figure(figsize=(6,3))
    ax2 = fig.add_subplot(111)
    ax2.imshow(log_spectrogram.T, aspect='auto', origin='lower', cmap=plt.cm.viridis, 
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax2.set_yticks(freqs[::20])
    ax2.set_xticks(times[::20])
    ax2.set_title('Normalized Log Spectrogram')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Time (s)')
    figfile3 = BytesIO()
    plt.savefig(figfile3, format='png')
    figfile3.seek(0)
    log_spectrogram_plot = base64.b64encode(figfile3.getvalue())
    return log_spectrogram_plot.decode('utf8')

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
    spectrogram_plot = None
    spectrogram_shape = None
    log_spectrogram_plot = None
    error_rate = None
    cv_similarity = None
    tfidf_similarity = None

    # Form for inference engine
    if form.validate_on_submit():
        partition = form.partition.data
        instance_number = form.instance_number.data
        # Get ground truth and predicted transcriptions
        truth_transcription = make_predictions.get_ground_truth(index=instance_number, partition=partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
        prediction_transcription = make_predictions.get_prediction(index=instance_number, partition=partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
        # Get features for visualizations
        vis_text, vis_spectrogram_feature, vis_audio_path, sample_rate, samples = make_predictions.vis_audio_features(index=instance_number, partition=partition)
        # Plot the audio waveform
        raw_plot = plot_raw_audio(sample_rate, samples)
        # Plot the spectrogram of the audio file
        spectrogram_plot = plot_spectrogram_feature(vis_spectrogram_feature)
        spectrogram_shape = 'The shape of the spectrogram of the chosen audio file: ' + str(vis_spectrogram_feature.shape)
        # 2nd and better plot of the spectrogram of the audio file
        freqs, times, log_spectrogram = log_spectrogram_feature(samples, sample_rate)
        mean = np.mean(log_spectrogram, axis=0)
        std = np.std(log_spectrogram, axis=0)
        log_spectrogram = (log_spectrogram - mean) / std
        log_spectrogram_plot = plot_log_spectrogram_feature(freqs, times, log_spectrogram)
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
    return render_template('index.html', title='Hey, Jetson!', form=form, truth_transcription=truth_transcription, prediction_transcription=prediction_transcription, raw_plot=raw_plot, spectrogram_plot=spectrogram_plot, log_spectrogram_plot=log_spectrogram_plot, spectrogram_shape=spectrogram_shape, error_rate=error_rate, cv_similarity=cv_similarity, tfidf_similarity=tfidf_similarity)
    
@app.route('/about')
@app.route('/about.html')
def about():
    spectrogram_3d = None
    vis_text, vis_spectrogram_feature, vis_audio_path, sample_rate, samples = make_predictions.vis_audio_features(index=95, partition='test')
    freqs, times, log_spectrogram = log_spectrogram_feature(samples, sample_rate)
    mean = np.mean(log_spectrogram, axis=0)
    std = np.std(log_spectrogram, axis=0)
    log_spectrogram = (log_spectrogram - mean) / std

    def plot_3d_spectrogram(log_spectrogram):
        data = [go.Surface(z=log_spectrogram.T, colorscale='Viridis')]
        layout = go.Layout(
        title='3D Spectrogram',
        scene = dict(
        yaxis = dict(title='Frequency', range=freqs),
        xaxis = dict(title='Time (s)', range=times),
        zaxis = dict(title='Log Amplitude'),),)
        fig = go.Figure(data=data, layout=layout)
        div_output = plot(fig, output_type='div', include_plotlyjs=False)
        return div_output

    spectrogram_3d = plot_3d_spectrogram(log_spectrogram)
    spectrogram_3d = Markup(spectrogram_3d)

    return render_template('about.html', title='Hey, Jetson!', spectrogram_3d=spectrogram_3d)

@app.route('/contact')
@app.route('/contact.html')
def contact():
    return render_template('contact.html', title='Hey, Jetson!')

app.secret_key = 'super_secret_key_shhhhhh'
if __name__ == '__main__':
    app.run(debug=True)
    app.run(ssl_context='adhoc')

#venv\Scripts\activate
#flask run --host 0.0.0.0
