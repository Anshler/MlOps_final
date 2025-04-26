import torch
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import accuracy_score
from safetensors.torch import load_file
from models import Model

# Load the model and test data
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None

try:
    MODEL_NAME = "bestModel"
    client = MlflowClient()
    production_model_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if production_model_versions:
        # Get the most recent production version (latest version with stage 'Production')
        latest_production_version = production_model_versions[0].version
        print(f"Latest production model version: {latest_production_version}")

        # Load the model from the model registry
        model_subpath = "model"
        model_uri = f"models:/{MODEL_NAME}/{latest_production_version}"
        model = mlflow.pytorch.load_model(model_uri)

        print("Model loaded successfully from registry")
    else:
        raise Exception(f"No production model found for {MODEL_NAME}. Using local backup instead")
except Exception as e:
    print(e)
    model = Model(hidden_layer=512).to(device)
    model.load_state_dict(load_file("local_prod_backup.safetensors", device=device))
    print("Backup model loaded successfully")

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
