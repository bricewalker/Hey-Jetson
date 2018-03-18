from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, DecimalField, IntegerField
from wtforms.validators import DataRequired, Required

from app import app
import make_predictions

class InferenceForm(FlaskForm):
    """Flask wtf Form to collect user input data"""
    partition = SelectField('Select data partition:', choices=[('test', 'test'), ('validation', 'validation')], coerce=str)
    instance_number = IntegerField('Enter individual instance number:', validators=[Required()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST', 'PUT'])
@app.route('/index', methods=['GET', 'POST', 'PUT'])
@app.route('/index.html', methods=['GET', 'POST', 'PUT'])
def index():
    form = InferenceForm()
    truth_transcription = None
    prediction_transcription = None
    if form.validate_on_submit():
        partition = form.partition.data
        instance_number = form.instance_number.data
        print(instance_number)
        print(partition)
        truth_transcription = make_predictions.get_ground_truth(index=instance_number, partition=partition, input_to_softmax=make_predictions.model_6, model_path='./results/model_6.h5')
        prediction_transcription = make_predictions.get_prediction(index=instance_number, partition=partition, input_to_softmax=make_predictions.model_6, model_path='./results/model_6.h5')
        print(truth_transcription)
        print(prediction_transcription)
    
    return render_template('index.html', title='Hey, Jetson!', form=form, truth_transcription=truth_transcription, prediction_transcription=prediction_transcription)

# partition=partition, instance_number=instance_number, truth_transcription=truth_transcription, prediction_transcription=prediction_transcription
# truth_transcription=truth_transcription, prediction_transcription=prediction_transcription
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