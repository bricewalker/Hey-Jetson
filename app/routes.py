from flask import render_template
from app import app
import make_predictions


@app.route('/')
@app.route('/index')
def index():
    truth_transcription = make_predictions.get_ground_truth(index=1000,     partition='test', input_to_softmax=make_predictions.model_6,   model_path='./results/model_6.h5')
    prediction_transcription = make_predictions.get_prediction(index=1000,     partition='test', input_to_softmax=make_predictions.model_6,   model_path='./results/model_6.h5')
    return render_template('index.html', title='Hey, Jetson!', truth_transcription=truth_transcription, prediction_transcription=prediction_transcription)

#@app.route('/about')
#def index():
#    return render_template('about.html', title='Hey, Jetson!')

#@app.route('/contact')
#def index():
#    return render_template('contact.html', title='Hey, Jetson!')