from flask import Flask, jsonify, request, render_template, session, redirect
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, DecimalField, IntegerField
from wtforms.validators import DataRequired, Required

from app import app
import make_predictions

class InferenceForm(FlaskForm):
    """Flask wtf Form to collect user input data"""
    param1 = SelectField('The data partition you would like to use:', choices=[('test', 'test'), ('validation', 'validation')], coerce=str)
    param2 = IntegerField('The individual instance you would like to run:', validators=[Required()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST', 'PUT'])
@app.route('/index', methods=['GET', 'POST', 'PUT'])
@app.route('/index.html', methods=['GET', 'POST', 'PUT'])
def index():
    print(session)
    form = InferenceForm()
    if form.validate_on_submit():
        session['partition'] = form.param1.data
        session['instance_number'] = form.param2.data
        session['truth_transcription'] = make_predictions.get_ground_truth(index=session['instance_number'], partition=session['partition'], input_to_softmax='make_predictions.model_6', model_path='./results/model_6.h5')
        session['prediction_transcription'] = make_predictions.get_prediction(index=session['instance_number'], partition=session['partition'], input_to_softmax='make_predictions.model_6', model_path='./results/model_6.h5')

        return redirect(url_for('index'))

    return render_template('index.html', title='Hey, Jetson!', form=form, **session)

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