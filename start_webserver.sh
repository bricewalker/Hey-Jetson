cd /home/brice/Hey-Jetson/
source venv/bin/activate
gunicorn -b localhost:8000 -w 1 inference:app
