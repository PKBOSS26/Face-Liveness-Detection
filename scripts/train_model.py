import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple model class
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes: live and spoof

    def forward(self, x):
        return self.model(x)

# Create a dummy dataset
def create_dummy_data(num_samples=100):
    X = torch.randn(num_samples, 3, 224, 224)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(X, y)

# Train the model (simple example)
def train_model():
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = create_dummy_data()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(2):  # Training for 2 epochs
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the model
    torch.save(model.state_dict(), 'models/model.pth')

if __name__ == '__main__':
    train_model()
