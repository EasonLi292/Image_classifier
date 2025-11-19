import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define a simple CNN model by extending nn.Module.
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # First convolution: 3 input channels, 64 output channels, 7x7 filter.
        # Padding is set to 3 so that the output image remains 32x32.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        
        # Second convolution: 64 input channels, 128 output channels, 3x3 filter.
        # Padding is 1 to keep spatial dimensions.
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Third block: two convolution layers.
        # First convolution in block: 128 input channels, 256 output channels.
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # Second convolution in block: 256 input channels, 256 output channels.
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        # Fully connected layer: from flattened feature map to 10 classes.
        # After three blocks with pooling, the image size reduces to 4x4,
        # and the number of channels is 256, so 4x4x256 = 4096.
        self.fc = nn.Linear(in_features=4096, out_features=10)
        
        # Max pooling layer: used in each block to reduce spatial dimensions.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Block 1: Convolution -> ReLU -> Max Pooling.
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)  # Now image size becomes 16x16.
        
        # Block 2: Convolution -> ReLU -> Max Pooling.
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)  # Now image size becomes 8x8.
        
        # Block 3: Two Convolutions -> ReLU -> Max Pooling.
        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.pool(x)  # Now image size becomes 4x4.
        
        # Flatten the feature map so that it can be fed into the fully connected layer.
        x = x.view(x.size(0), -1)  # x.size(0) is the batch size.
        x = self.fc(x)  # This outputs 10 values, one for each class.
        return x

def main():
    # Check if a GPU is available; otherwise use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Define the transformation for the images:
    # Convert images to tensors and normalize pixel values.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load the CIFAR-10 training and test datasets.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create data loaders to iterate over the datasets in batches.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    
    # Create an instance of the model and move it to the chosen device.
    model = BasicCNN().to(device)
    
    # Define the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification.
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer.
    
    # Variables to store training time and errors for each epoch.
    train_times = []
    train_errors = []
    test_errors = []
    num_epochs = 10  # You can change the number of epochs if desired.
    
    # Training loop.
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode.
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time = time.time()
        
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Clear the gradients.
            optimizer.zero_grad()
            
            # Forward pass.
            outputs = model(images)
            
            # Calculate the loss.
            loss = criterion(outputs, labels)
            
            # Backward pass to compute gradients.
            loss.backward()
            
            # Update the model parameters.
            optimizer.step()
            
            # Update running loss and training accuracy.
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Calculate the time taken for the epoch.
        epoch_time = time.time() - start_time
        train_times.append(epoch_time)
        
        # Calculate the training error for this epoch.
        train_error = 1 - correct_train / total_train
        train_errors.append(train_error)
        
        # Evaluate the model on the test dataset.
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        test_error = 1 - correct_test / total_test
        test_errors.append(test_error)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.2f}s, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")
    
    # Plot training time per epoch.
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_times, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time per Epoch')
    plt.show()
    
    # Plot training and testing errors.
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_errors, marker='o', label='Training Error')
    plt.plot(range(1, num_epochs+1), test_errors, marker='o', label='Testing Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training & Testing Errors per Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()