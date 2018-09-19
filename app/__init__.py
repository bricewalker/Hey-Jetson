# This initializes the flask web app

from flask import Flask

app = Flask(__name__)

from app import routes
