from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')

# Load the model and vectorizer
print("Loading the saved model and vectorizer...")
model = joblib.load('backend/model.pkl')
vectorizer = joblib.load('backend/vectorizer.pkl')
print("Model and vectorizer loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get text input from the request
    data = request.json
    print(f"Received data: {data}")  # Log incoming data
    text = data.get('text', '')

    # Check if text is provided
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)

        # Convert prediction to a native Python int
        prediction_value = int(prediction[0])  # Convert from int64 to int

        # Map the prediction value to a string
        prediction_label = "Human Written" if prediction_value == 0 else "AI Generated"
        print(f"Raw prediction: {prediction_value}")  # Debugging line


        # Return prediction as JSON
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log the error for debugging
        return jsonify({'error': 'An error occurred during prediction. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
