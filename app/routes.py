# This is the primary script for defining the routes for the flask web app

# Flask package imports
from flask import Flask, jsonify, request, render_template, session, redirect, url_for, send_from_directory, Markup, abort
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import StringField, SubmitField, SelectField, IntegerField
from wtforms.validators import DataRequired, Required
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict

# Common, File Based, and Math Imports
import pandas as pd
import numpy as np
import collections
import os
from os.path import isdir, join
from pathlib import Path
import subprocess
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
import timeit
import time
import base64
import datetime
import uuid
import wave
import requests
import audioop
from io import BytesIO

# Audio processing imports
from scipy import signal
from scipy.fftpack import dct
import soundfile
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from scipy.fftpack import fft

# Model metric imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from flask import Flask, render_template, send_file, make_response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Visualization settings
color = sns.color_palette()
sns.set_style('darkgrid')

# Script imports
from app import app
import make_predictions

# Setting random seeds
np.random.seed(95)
RNG_SEED = 95

from azure.keyvault import KeyVaultClient, KeyVaultAuthentication
from azure.common.credentials import ServicePrincipalCredentials

credentials = None

AZURE_TENANT_ID = '89dd8617-dd08-49d6-a072-bf0b4cc27084'
AZURE_CLIENT_ID = '0db1e353-1aa2-4bc2-838d-7511578ca2bd'
AZURE_CLIENT_OID = '18be8686-9958-4cf5-9ccd-cde327077606'
AZURE_CLIENT_SECRET = 'h81+5KGSgbcazglgQNAvV9voor6SmDABW79km97aZrk='
AZURE_SUBSCRIPTION_ID = '397f6ac2-359d-4e8f-9712-eb8f4930dfe6'

def auth_callback(server, resource, scope):
    credentials = ServicePrincipalCredentials(
        client_id = AZURE_CLIENT_ID,
        secret = AZURE_CLIENT_SECRET,
        tenant = AZURE_TENANT_ID,
        resource = "https://vault.azure.net"
    )
    token = credentials.token
    return token['token_type'], token['access_token']

client = KeyVaultClient(KeyVaultAuthentication(auth_callback))

secret_bundle = client.get_secret("https://VAULT_ID.vault.azure.net/", "SECRET_ID", "SECRET_VERSION")

print(secret_bundle.value)

# Microsoft Cognitive Services Speech API parameters
SUBSCRIPTION_KEY = 'YOUR_AZURE_SPEECH_API_KEY'
assert SUBSCRIPTION_KEY
headers = {
           'Transfer-Encoding': 'chunked',
           'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
           'Content-type': 'audio/wav; codec=audio/pcm; samplerate=16000'
          }

params = (('language', 'en-us'), ('format', 'detailed'),)

# Microsoft Cognitive Services Text Analytics API parameters
SUB_KEY = 'YOUR_AZURE_TEXT_API_KEY'
assert SUB_KEY
text_analytics_base_url = "https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/"
language_api_url = text_analytics_base_url + "languages"
sentiment_api_url = text_analytics_base_url + "sentiment"
key_phrase_api_url = text_analytics_base_url + "keyPhrases"
entity_linking_api_url = text_analytics_base_url + "entities"
text_headers   = {"Ocp-Apim-Subscription-Key": SUB_KEY}
sentiment_api_url = text_analytics_base_url + "sentiment"

# Flask wtform to collect user input data for inference engine
class AudioForm(FlaskForm):
    audio_file = FileField(validators=[FileRequired()])

# Flask wtform to collect user input data for visualization engine
class VisualizationForm(FlaskForm):
    viz_model_number = SelectField('Select model:', choices=[('model_10', 'model_10'), ('model_8', 'model_8')], coerce=str)
    viz_partition = SelectField('Select data partition:', choices=[('validation', 'validation'), ('test', 'test')], coerce=str)
    viz_instance_number = IntegerField('Enter individual instance number:', validators=[Required()])
    viz_submit = SubmitField('Submit')

# Flask wtform to collect user input data for performance engine
class PerformanceForm(FlaskForm):
    perf_model_number = SelectField('Select model:', choices=[('model_10', 'model_10'), ('model_8', 'model_8')], coerce=str)
    perf_partition = SelectField('Select data partition:', choices=[('validation', 'validation'), ('test', 'test')], coerce=str)
    perf_instance_number = IntegerField('Enter individual instance number:', validators=[Required()])
    perf_submit = SubmitField('Submit')

# Flask wtform to collect user input data for sentiment engine
class SentimentForm(FlaskForm):
    sent_model_number = SelectField('Select model:', choices=[('model_10', 'model_10'), ('model_8', 'model_8')], coerce=str)
    sent_partition = SelectField('Select data partition:', choices=[('validation', 'validation'), ('test', 'test')], coerce=str)
    sent_instance_number = IntegerField('Enter individual instance number:', validators=[Required()])
    sent_submit = SubmitField('Submit')

@app.after_request
def add_header(r):
    # Add headers to both force latest rendering engine and prevent cache
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    # Initializing form for user input
    audio_form = AudioForm(CombinedMultiDict((request.files, request.form)))
    # Initializing variables passed to HTML files
    filename = None
    prediction_transcription = None
    raw_plot = None
    spectrogram_plot = None
    spectrogram_shape = None
    log_spectrogram_plot = None
    spectrogram_3d = None
    word_error_rate = None
    cv_similarity = None
    jetson_time_to_predict = None
    cortana_time_to_predict = None
    cortana_transcription = None
    recognitionstatus = None
    offset = None
    duration = None
    nbest = None
    confidence = None
    lexical = None
    itn = None
    maskeditn = None
    display = None
    sentiments = None
    documents = None
    errors = None
    prediction_score = None
    prediction_id = None
    cortana_score = None
    cortana_id = None

    # Form for inference engine
    if audio_form.validate_on_submit():
        f = audio_form.audio_file.data
        filename = os.path.join('app/static/audio/', 'tmp.wav')
        f.save(filename)
        # Connecting to Microsoft Speech API for Cortana's predicted transcription 
        c_start = time.time()
        audiofile =  open(filename, 'rb')
        response = requests.post('https://westus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1', headers=headers, params=params, data=make_predictions.read_in_chunks(audiofile))
        cortana_transcription = response.content
        c_end = time.time()
        cortana_time_to_predict = c_end - c_start
        val = json.loads(response.text)
        recognitionstatus = val["RecognitionStatus"]
        offset = val["Offset"]
        duration = val["Duration"]
        nbest = val["NBest"]
        confidence = val["NBest"][0]["Confidence"]
        lexical = val["NBest"][0]["Lexical"]
        itn = val["NBest"][0]["ITN"]
        maskeditn = val["NBest"][0]["MaskedITN"]
        display = val["NBest"][0]["Display"]
        # Producing Hey, Jetson! predicted transcription
        s_start = time.time()
        prediction_transcription = make_predictions.run_inference(audio_path=filename, input_to_softmax=make_predictions.model_10, model_path='./results/model_10.h5')
        s_end = time.time()
        jetson_time_to_predict = s_end - s_start
        vis_spectrogram_feature, sample_rate, samples = make_predictions.inference_vis_audio_features(index=filename)
        # Plot the audio waveform
        raw_plot = make_predictions.plot_raw_audio(sample_rate, samples)
        # Plot the spectrogram of the audio file
        spectrogram_plot = make_predictions.plot_spectrogram_feature(vis_spectrogram_feature)
        spectrogram_shape = 'The shape of the spectrogram of the uploaded audio file: ' + str(vis_spectrogram_feature.shape)
        # 2nd way to plot the spectrogram of the audio file
        freqs, times, log_spectrogram = make_predictions.log_spectrogram_feature(samples, sample_rate)
        mean = np.mean(log_spectrogram, axis=0)
        std = np.std(log_spectrogram, axis=0)
        log_spectrogram = (log_spectrogram - mean) / std
        log_spectrogram_plot = make_predictions.plot_log_spectrogram_feature(freqs, times, log_spectrogram)
        # 3d plot of the spectrogram of a random audio file from the test set, plotting amplitude over frequency over time.
        def plot_3d_spectrogram(log_spectrogram):
            data = [go.Surface(z=log_spectrogram.T, colorscale='Viridis')]
            layout = go.Layout(
            title='3D Spectrogram',
            autosize=True,
            width=700,
            height=700,
            margin=dict(l=50, r=50, b=50, t=50))
            fig = go.Figure(data=data, layout=layout)
            div_output = plot(fig, output_type='div', include_plotlyjs=False)
            return div_output
        # 3d spectrogram plot
        spectrogram_3d = plot_3d_spectrogram(log_spectrogram)
        spectrogram_3d = Markup(spectrogram_3d)
        # Connecting to Microsoft Text Analytics API for sentiment analysis
        text_documents = {'documents' : [{'id': 'Predicted Transcription', 'language': 'en', 'text': prediction_transcription},
                        {'id': 'Cortana Transcription', 'language': 'en', 'text': lexical}
                    ]}
        sentiment_response  = requests.post(sentiment_api_url, headers=text_headers, json=text_documents)
        sentiments = sentiment_response.json()
        documents = sentiments["documents"]
        errors = sentiments["errors"]
        prediction_score = sentiments["documents"][0]["score"]
        prediction_id = sentiments["documents"][0]["id"]
        cortana_score = sentiments["documents"][1]["score"]
        cortana_id = sentiments["documents"][1]["id"]

    # Render the html page.
    return render_template('index.html', audio_form=audio_form, filename=filename, prediction_transcription=prediction_transcription, 
                           raw_plot=raw_plot, spectrogram_plot=spectrogram_plot, log_spectrogram_plot=log_spectrogram_plot, 
                           spectrogram_shape=spectrogram_shape, spectrogram_3d=spectrogram_3d, jetson_time_to_predict=jetson_time_to_predict, 
                           cortana_time_to_predict=cortana_time_to_predict, confidence=confidence, lexical=lexical, itn=itn, maskeditn=maskeditn, display=display, prediction_score=prediction_score, cortana_score=cortana_score)

@app.route('/asr', methods=['GET', 'POST'])
@app.route('/asr.html', methods=['GET', 'POST'])
def asr():
    # Render the html page.
    return render_template('asr.html')

@app.route('/visualization', methods=['GET', 'POST'])
@app.route('/visualization.html', methods=['GET', 'POST'])
def visualization():
    # Initializing form for user input
    visualization_form = VisualizationForm()
    # Initializing variables passed to HTML files
    truth_transcription = None
    prediction_transcription = None
    raw_plot = None
    spectrogram_plot = None
    spectrogram_shape = None
    log_spectrogram_plot = None
    spectrogram_3d = None
    cortana_transcription = None
    recognitionstatus = None
    offset = None
    duration = None
    nbest = None
    confidence = None
    lexical = None
    itn = None
    maskeditn = None
    display = None
    play_audio = None

    # Form for visualization engine
    if visualization_form.validate_on_submit():
        v_model_number = visualization_form.viz_model_number.data
        v_partition = visualization_form.viz_partition.data
        v_instance_number = visualization_form.viz_instance_number.data
        # Get ground truth and predicted transcriptions
        if v_model_number == 'model_10':
            truth_transcription = make_predictions.get_ground_truth(index=v_instance_number, partition=v_partition, input_to_softmax=make_predictions.model_10, model_path='./results/model_10.h5')
            prediction_transcription = make_predictions.get_prediction(index=v_instance_number, partition=v_partition, input_to_softmax=make_predictions.model_10, model_path='./results/model_10.h5')
        else:
            truth_transcription = make_predictions.get_ground_truth(index=v_instance_number, partition=v_partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
            prediction_transcription = make_predictions.get_prediction(index=v_instance_number, partition=v_partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
        # Get features for visualizations
        vis_text, vis_spectrogram_feature, vis_audio_path, sample_rate, samples = make_predictions.vis_audio_features(index=v_instance_number, partition=v_partition)
        # Plot the audio waveform
        raw_plot = make_predictions.plot_raw_audio(sample_rate, samples)
        # Plot the spectrogram of the audio file
        spectrogram_plot = make_predictions.plot_spectrogram_feature(vis_spectrogram_feature)
        spectrogram_shape = 'The shape of the spectrogram of the chosen audio file: ' + str(vis_spectrogram_feature.shape)
        # 2nd way to plot the spectrogram of the audio file
        freqs, times, log_spectrogram = make_predictions.log_spectrogram_feature(samples, sample_rate)
        mean = np.mean(log_spectrogram, axis=0)
        std = np.std(log_spectrogram, axis=0)
        log_spectrogram = (log_spectrogram - mean) / std
        log_spectrogram_plot = make_predictions.plot_log_spectrogram_feature(freqs, times, log_spectrogram)
        # 3d plot of the spectrogram of a random audio file from the test set, plotting amplitude over frequency over time.
        def plot_3d_spectrogram(log_spectrogram):
            data = [go.Surface(z=log_spectrogram.T, colorscale='Viridis')]
            layout = go.Layout(
            title='3D Spectrogram',
            autosize=True,
            width=700,
            height=700,
            margin=dict(l=50, r=50, b=50, t=50))
            fig = go.Figure(data=data, layout=layout)
            div_output = plot(fig, output_type='div', include_plotlyjs=False)
            return div_output
        # 3d spectrogram plot
        spectrogram_3d = plot_3d_spectrogram(log_spectrogram)
        spectrogram_3d = Markup(spectrogram_3d)
        # Connecting to Microsoft Speech API for Cortana's predicted transcription
        filepath = make_predictions.azure_inference(index=v_instance_number, partition=v_partition)
        audiofile =  open(filepath, 'rb')
        response = requests.post('https://westus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1',
                                 headers=headers, params=params, data=make_predictions.read_in_chunks(audiofile))
        cortana_transcription = response.content
        val = json.loads(response.text)
        recognitionstatus = val["RecognitionStatus"]
        offset = val["Offset"]
        duration = val["Duration"]
        nbest = val["NBest"]
        confidence = val["NBest"][0]["Confidence"]
        lexical = val["NBest"][0]["Lexical"]
        itn = val["NBest"][0]["ITN"]
        maskeditn = val["NBest"][0]["MaskedITN"]
        display = val["NBest"][0]["Display"]
        # Serve the audio file for the audio player
        play_audio = filepath.replace("/home/brice/Hey-Jetson/app/", "")

    # Render the html page.
    return render_template('visualization.html', visualization_form=visualization_form, truth_transcription=truth_transcription, prediction_transcription=prediction_transcription, raw_plot=raw_plot, spectrogram_plot=spectrogram_plot, log_spectrogram_plot=log_spectrogram_plot, spectrogram_shape=spectrogram_shape, spectrogram_3d=spectrogram_3d, cortana_transcription=cortana_transcription, confidence=confidence, lexical=lexical, itn=itn, maskeditn=maskeditn, display=display, play_audio=play_audio)

@app.route('/performance', methods=['GET', 'POST'])
@app.route('/performance.html', methods=['GET', 'POST'])
def performance():
    # Initializing form for user input
    performance_form = PerformanceForm()
    # Initializing variables passed to HTML files
    truth_transcription = None
    prediction_transcription = None
    word_error_rate = None
    cv_similarity = None
    tfidf_similarity = None
    jetson_time_to_predict = None
    cortana_time_to_predict = None
    cortana_transcription = None
    recognitionstatus = None
    offset = None
    duration = None
    nbest = None
    confidence = None
    lexical = None
    itn = None
    maskeditn = None
    display = None
    cortana_cv = None
    cortana_tfidf = None
    cortana_wer = None
    play_audio = None

    # Form for performance engine
    if performance_form.validate_on_submit():
        p_model_number = performance_form.perf_model_number.data
        p_partition = performance_form.perf_partition.data
        p_instance_number = performance_form.perf_instance_number.data
        # Get ground truth and predicted transcriptions
        if p_model_number == 'model_10':
            truth_transcription = make_predictions.get_ground_truth(index=p_instance_number, partition=p_partition, input_to_softmax=make_predictions.model_10, model_path='./results/model_10.h5')
            start = time.time()
            prediction_transcription = make_predictions.get_prediction(index=p_instance_number, partition=p_partition, input_to_softmax=make_predictions.model_10, model_path='./results/model_10.h5')
            end = time.time()
            jetson_time_to_predict = end - start
        else:
            truth_transcription = make_predictions.get_ground_truth(index=p_instance_number, partition=p_partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
            start = time.time()
            prediction_transcription = make_predictions.get_prediction(index=p_instance_number, partition=p_partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
            end = time.time()
            jetson_time_to_predict = end - start
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
        # Calculate word error rate of individual transcription
        word_error_rate = make_predictions.wer_calc(truth_transcription, prediction_transcription)
        # Connecting to Microsoft Speech API for Cortana's predicted transcription
        c_start = time.time()
        filepath = make_predictions.azure_inference(index=p_instance_number, partition=p_partition)
        audiofile =  open(filepath, 'rb')
        response = requests.post('https://westus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1',
                                 headers=headers, params=params, data=make_predictions.read_in_chunks(audiofile))
        cortana_transcription = response.content
        c_end = time.time()
        cortana_time_to_predict = c_end - c_start
        val = json.loads(response.text)
        recognitionstatus = val["RecognitionStatus"]
        offset = val["Offset"]
        duration = val["Duration"]
        nbest = val["NBest"]
        confidence = val["NBest"][0]["Confidence"]
        lexical = val["NBest"][0]["Lexical"]
        itn = val["NBest"][0]["ITN"]
        maskeditn = val["NBest"][0]["MaskedITN"]
        display = val["NBest"][0]["Display"]
        # Calculate performance measures on AZURE transcript
        cv_cortana_vec = cv.transform([lexical])
        cortana_cv = cosine_similarity(cv_ground_truth_vec, cv_cortana_vec)
        tfidf_cortana_vec = tfidf.transform([lexical])
        cortana_tfidf = cosine_similarity(tfidf_ground_truth_vec, tfidf_cortana_vec)
        cortana_wer = make_predictions.wer_calc(truth_transcription, lexical)
        # Serve the audio file for the audio player
        play_audio = filepath.replace("/home/brice/Hey-Jetson/app/", "")
    
    # Render the html page
    return render_template('performance.html', performance_form=performance_form, truth_transcription=truth_transcription, prediction_transcription=prediction_transcription, word_error_rate=word_error_rate, cv_similarity=cv_similarity, tfidf_similarity=tfidf_similarity, jetson_time_to_predict=jetson_time_to_predict, cortana_transcription=cortana_transcription, cortana_time_to_predict=cortana_time_to_predict, confidence=confidence, lexical=lexical, itn=itn, maskeditn=maskeditn, display=display, cortana_cv=cortana_cv, cortana_tfidf=cortana_tfidf, cortana_wer=cortana_wer, play_audio=play_audio)

@app.route('/sentiment', methods=['GET', 'POST'])
@app.route('/sentiment.html', methods=['GET', 'POST'])
def sentiment():
    # Initializing form for user input
    sentiment_form = SentimentForm()
    # Initializing variables passed to HTML files
    truth_transcription = None
    prediction_transcription = None
    cortana_transcription = None
    recognitionstatus = None
    offset = None
    duration = None
    nbest = None
    confidence = None
    lexical = None
    itn = None
    maskeditn = None
    display = None
    sentiments = None
    documents = None
    errors = None
    truth_score = None
    truth_id = None
    prediction_score = None
    prediction_id = None
    cortana_score = None
    cortana_id = None
    play_audio = None

    # Form for sentiment engine
    if sentiment_form.validate_on_submit():
        s_model_number = sentiment_form.sent_model_number.data
        s_partition = sentiment_form.sent_partition.data
        s_instance_number = sentiment_form.sent_instance_number.data
        # Get ground truth and predicted transcriptions
        if s_model_number == 'model_10':
            truth_transcription = make_predictions.get_ground_truth(index=s_instance_number, partition=s_partition, input_to_softmax=make_predictions.model_10, model_path='./results/model_10.h5')
            prediction_transcription = make_predictions.get_prediction(index=s_instance_number, partition=s_partition, input_to_softmax=make_predictions.model_10, model_path='./results/model_10.h5')
        else:
            truth_transcription = make_predictions.get_ground_truth(index=s_instance_number, partition=s_partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
            prediction_transcription = make_predictions.get_prediction(index=s_instance_number, partition=s_partition, input_to_softmax=make_predictions.model_8, model_path='./results/model_8.h5')
        # Connecting to Microsoft Speech API for Cortana's predicted transcription
        filepath = make_predictions.azure_inference(index=s_instance_number, partition=s_partition)
        audiofile =  open(filepath, 'rb')
        response = requests.post('https://westus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1',
                                 headers=headers, params=params, data=make_predictions.read_in_chunks(audiofile))
        cortana_transcription = response.content
        val = json.loads(response.text)
        recognitionstatus = val["RecognitionStatus"]
        offset = val["Offset"]
        duration = val["Duration"]
        nbest = val["NBest"]
        confidence = val["NBest"][0]["Confidence"]
        lexical = val["NBest"][0]["Lexical"]
        itn = val["NBest"][0]["ITN"]
        maskeditn = val["NBest"][0]["MaskedITN"]
        display = val["NBest"][0]["Display"]
        # Connecting to Microsoft Text Analytics API for sentiment analysis
        text_documents = {'documents' : [
                        {'id': 'Ground Truth Transcription', 'language': 'en', 'text': truth_transcription},
                        {'id': 'Predicted Transcription', 'language': 'en', 'text': prediction_transcription},
                        {'id': 'Cortana Transcription', 'language': 'en', 'text': lexical}
                    ]}
        sentiment_response  = requests.post(sentiment_api_url, headers=text_headers, json=text_documents)
        sentiments = sentiment_response.json()
        documents = sentiments["documents"]
        errors = sentiments["errors"]
        truth_score = sentiments["documents"][0]["score"]
        truth_id = sentiments["documents"][0]["id"]
        prediction_score = sentiments["documents"][1]["score"]
        prediction_id = sentiments["documents"][1]["id"]
        cortana_score = sentiments["documents"][2]["score"]
        cortana_id = sentiments["documents"][2]["id"]
        # Serve the audio file for the audio player
        play_audio = filepath.replace("/home/brice/Hey-Jetson/app/", "")
    
    # Render the html page
    return render_template('sentiment.html', sentiment_form=sentiment_form, truth_transcription=truth_transcription, prediction_transcription=prediction_transcription, cortana_transcription=cortana_transcription, confidence=confidence, lexical=lexical, itn=itn, maskeditn=maskeditn, display=display, truth_score=truth_score, prediction_score=prediction_score, cortana_score=cortana_score, play_audio=play_audio)
    
@app.route('/about')
@app.route('/about.html')
def about():
    # Initializing variables passed to HTML files
    spectrogram_3d = None
    # Creating variables for 3d spectrogram plot
    vis_text, vis_spectrogram_feature, vis_audio_path, sample_rate, samples = make_predictions.vis_audio_features(index=np.random.randint(0, 4176), partition='test')
    freqs, times, log_spectrogram = make_predictions.log_spectrogram_feature(samples, sample_rate)
    mean = np.mean(log_spectrogram, axis=0)
    std = np.std(log_spectrogram, axis=0)
    log_spectrogram = (log_spectrogram - mean) / std
    # 3d plot of the spectrogram of a random audio file from the test set, plotting amplitude over frequency over time.
    def plot_3d_spectrogram(log_spectrogram):
        data = [go.Surface(z=log_spectrogram.T, colorscale='Viridis')]
        layout = go.Layout(
        title='3D Spectrogram',
        autosize=True,
        width=700,
        height=700,
        margin=dict(l=50, r=50, b=50, t=50))
        fig = go.Figure(data=data, layout=layout)
        div_output = plot(fig, output_type='div', include_plotlyjs=False)
        return div_output
    # Converting 3d plot for JavaScript rendering
    spectrogram_3d = plot_3d_spectrogram(log_spectrogram)
    spectrogram_3d = Markup(spectrogram_3d)
    # render the HTML page
    return render_template('about.html', spectrogram_3d=spectrogram_3d)

@app.route('/contact')
@app.route('/contact.html')
def contact():
    # Render the html page
    return render_template('contact.html')

app.secret_key = 'super_secret_key_shhhhhh'
if __name__ == '__main__':
    app.run(debug=True)
    app.run(ssl_context='adhoc')