document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const resultDiv = document.getElementById('result');

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the form from submitting the traditional way
        const formData = new FormData(uploadForm);

        fetch('/detect_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Display the prediction result
            resultDiv.innerHTML = 'Prediction: ' + data.prediction;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
