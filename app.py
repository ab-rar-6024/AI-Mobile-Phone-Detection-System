from flask import Flask, render_template, Response
import os
from detector import generate_frames

app = Flask(__name__)

@app.route('/')
def index():
    folder = "static/detections"
    images = os.listdir(folder)
    images = sorted(images, reverse=True)
    return render_template('index.html', images=images)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)