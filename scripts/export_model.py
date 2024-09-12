import torch
import torch.nn as nn
import torch.onnx
import torchvision.models as models

# Define the model class
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes: live and spoof

    def forward(self, x):
        return self.model(x)

# Load the trained model
def load_model(model_path='models/model.pth'):
    model = SimpleNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Export the model to ONNX format
def export_model():
    model = load_model()
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input tensor
    torch.onnx.export(model, dummy_input, 'public/model.onnx', verbose=True, input_names=['input'], output_names=['output'])

if __name__ == '__main__':
    export_model()
