document.getElementById('submitBtn').addEventListener('click', async () => {
    const textInput = document.getElementById('textInput').value;

    // Validate input
    if (!textInput.trim()) {
        alert("Please enter some text.");
        return;  // Exit the function if the input is empty
    }

    // Send POST request to /predict endpoint
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: textInput }),
    });

    const resultElement = document.getElementById('result');

    if (response.ok) {
        const data = await response.json();
        resultElement.innerText = `Prediction: ${data.prediction}`;
    } else {
        const errorData = await response.json();
        resultElement.innerText = `Error: ${errorData.error}`;
    }
});
