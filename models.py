import torch
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch import nn
from safetensors.torch import load_file

class Model(nn.Module):
    def __init__(self, hidden_layer:int = 64):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(20, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, hidden_layer)
        self.fc4 = nn.Linear(hidden_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

def load_model(device:str = "cuda", model_name:str = "bestModel") -> nn.Module:
    try:
        client = MlflowClient()
        production_model_versions = client.get_latest_versions(model_name, stages=["Production"])

        if production_model_versions:
            # Get the most recent production version (latest version with stage 'Production')
            latest_production_version = production_model_versions[0].version
            print(f"Latest production model version: {latest_production_version}")

            # Load the model from the model registry
            model_uri = f"models:/{model_name}/{latest_production_version}"
            model = mlflow.pytorch.load_model(model_uri)

            print("Model loaded successfully from registry")
        else:
            raise Exception(f"No production model found for {model_name}. Using local backup instead")
    except Exception as e:
        print(e)
        model = Model(hidden_layer=512).to(device)
        model.load_state_dict(load_file("local_prod_backup.safetensors", device=device))

        print("Backup model loaded successfully")

    model.eval()
    return model