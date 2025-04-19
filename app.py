import torch
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import accuracy_score
from safetensors.torch import load_file
from models import Model

# Load the model and test data
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model(hidden_layer=512).to(device)
model.load_state_dict(load_file("best_model.safetensors", device=device))
model.eval()

test_data = torch.load("test_data.pt")
x_test_tensor = test_data['X_test'].to(device)
y_test_tensor = test_data['y_test'].to(device)
total_samples = x_test_tensor.shape[0]

# Initialize Flask app
app = Flask(__name__)


# Home route to display the UI
@app.route('/')
def index():
    return render_template('index.html')


# Route to sample test data and make predictions
@app.route('/generate', methods=['POST'])
def generate_and_predict():
    # user input
    n_samples = int(request.form['n_samples'])

    indices = torch.randperm(total_samples)[:n_samples]

    with torch.no_grad():
        y_pred_test = model(x_test_tensor[indices]).squeeze()
        y_pred_test = (y_pred_test > 0.5).float()

    accuracy = accuracy_score(y_test_tensor[indices].cpu(), y_pred_test.cpu())
    return jsonify({'predictions': y_pred_test.cpu().tolist(), 'labels': y_test_tensor[indices].cpu().tolist(), 'accuracy': accuracy})


if __name__ == '__main__':
    app.run(debug=True)
