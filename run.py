from flask import Flask, render_template
from ai_human_text_detector.backend.app import app as text_detector_app
from ai_human_image_detector.backend.app import app as image_detector_app

# Create the main Flask application
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Register blueprints for text and image detection
app.register_blueprint(text_detector_app, url_prefix='/text-detection')
app.register_blueprint(image_detector_app, url_prefix='/image-detection')

if __name__ == '__main__':
    app.run(debug=True)
