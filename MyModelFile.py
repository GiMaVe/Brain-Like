import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import tqdm
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import torch
import torch.nn as nn
import torch.nn.functional as F

def log(message):
    print(message)
    

def calculate_feature_size(model, input_shape=(1, 3, 224, 224)):
    # Create a dummy input tensor with the specified input shape
    input_tensor = torch.randn(input_shape)
    # Forward pass through the model up to the max pooling layer
    with torch.no_grad():  # Ensure no gradients are calculated
        output = model.features(input_tensor)
        output = model.maxpool(output)
    # Flatten the output and return its size
    output_flattened = output.view(output.size(0), -1)
    return output_flattened.size(1)




class CustomResNet(nn.Module):
    def __init__(self, pretrained_model, dropout_rate, num_neurons1, num_neurons2, num_neurons3, output_neurons=168):
        super(CustomResNet, self).__init__()
        # Using only the first seven layers of the pretrained model as feature extractor
        self.features = nn.Sequential(*list(pretrained_model.children())[:7])
        # Additional max pooling to reduce feature dimensions
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the input features to the first fully connected layer
        input_features = self.calculate_feature_size(input_shape=(1, 3, 224, 224))
        
        # Define the dense layers with batch normalization and ReLU activation
        self.fc1 = nn.Linear(input_features, num_neurons1)
        self.bn1 = nn.BatchNorm1d(num_neurons1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(num_neurons1, num_neurons2)
        self.bn2 = nn.BatchNorm1d(num_neurons2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(num_neurons2, num_neurons2)
        self.bn3 = nn.BatchNorm1d(num_neurons2)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(num_neurons2, output_neurons)

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.output(x)
        return x

    def calculate_feature_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(input_shape)
            output = self.features(self.maxpool(dummy_input))
            return output.numel()


# Define the objective function for Optuna
def objective(trial):
    # Start of trial
    print(f"Starting trial {trial.number} with hyperparameters.")

    # Hyperparameters to tune
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    num_neurons1 = trial.suggest_int('num_neurons1', 1024, 2048, step=128)
    num_neurons2 = trial.suggest_int('num_neurons2', 512, 1024, step=128)
    num_neurons3 = trial.suggest_int('num_neurons3', 256, 768, step=128)
    weight_decay = trial.suggest_loguniform('l2weightdecay', 1e-4, 1e-2)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)

    # Load the pretrained ResNet model
    base_model = models.resnet50(pretrained=True)
    for param in base_model.parameters():
        param.requires_grad = False

    # Instantiate the model
    print('Check1')
    model = CustomResNet(base_model, dropout_rate, num_neurons1, num_neurons2, num_neurons3)

    # Optimizer and loss
    print('Check2')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Data loaders
    print('Check3')
    train_dataloader = DataLoader(train_ds, batch_size=48, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1000, shuffle=False)

    # Training loop
    model.train()
    for epoch in range(3):  # Use a small number of epochs for trial
        epoch_loss = 0
        print('CheckEpoch')
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Debugging at the end of each epoch
        print(f"Epoch {epoch+1}/{3}, Loss: {epoch_loss / len(train_dataloader)}")

    # Validation loop
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    average_loss = total_loss / len(val_dataloader)
    # Final debugging at the end of a trial
    print(f"Trial {trial.number} completed. Validation loss: {average_loss}")
    return average_loss

