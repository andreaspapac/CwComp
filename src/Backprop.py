import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from Datasets import *

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
data_path = 'C:/Users/Andreas/Desktop/PhD/NeurIPS_Re/src/data/'
batch_size = 128
# # Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root=data_path, train=False, transform=transform_test, download=True)

# Load CIFAR-100 dataset
# train_dataset = CIFAR100(root='data/', train=True, transform=transform_train, download=True)
# test_dataset = CIFAR100(root='data/', train=False, transform=transform_test, download=True)

train_loader_CIFAR = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader_CIFAR = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# # Define transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# # Load datasets
# train_dataset_mnist = MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset_mnist = MNIST(root='./data', train=False, download=True, transform=transform)
#
# train_dataset_fashion = FashionMNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset_fashion = FashionMNIST(root='./data', train=False, download=True, transform=transform)
#
# # Define dataloaders
# train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=128, shuffle=True)
# test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=128, shuffle=False)
#
# train_loader_fashion = DataLoader(train_dataset_fashion, batch_size=128, shuffle=True)
# test_loader_fashion = DataLoader(test_dataset_fashion, batch_size=128, shuffle=False)

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 80, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 320, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(320)
        self.fc1 = nn.Linear(20480, 10)
        # self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 320 * 7 * 7)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(160)
        self.conv2 = nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1, groups=1)
        self.bn2 = nn.BatchNorm2d(160)
        self.conv3 = nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(160)
        self.conv4 = nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1, groups=1)
        self.bn4 = nn.BatchNorm2d(160)
        # self.fc1 = nn.Linear(30720, 10)

        self.fc1 = nn.Linear(10240, 10)
        # self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        return x

class CIF100Net(nn.Module):
    def __init__(self):
        super(CIF100Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 60, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(60)
        self.conv2 = nn.Conv2d(60, 120, kernel_size=3, stride=1, padding=1, groups=20)
        self.bn2 = nn.BatchNorm2d(120)
        self.conv3 = nn.Conv2d(120, 240, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(240)
        self.conv4 = nn.Conv2d(240, 400, kernel_size=3, stride=1, padding=1, groups=20)
        self.bn4 = nn.BatchNorm2d(400)
        self.conv5 = nn.Conv2d(400, 800, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(800)
        self.conv6 = nn.Conv2d(800, 1600, kernel_size=3, stride=1, padding=1, groups=100)
        self.bn6 = nn.BatchNorm2d(1600)
        # self.fc1 = nn.Linear(30720, 10)

        self.fc1 = nn.Linear(25600, 100)
        # self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        return x

# Define training and evaluation functions
def train(model, dataloader, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item() * data.size(0)
        predicted = output.argmax(dim=1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    return train_loss, accuracy

def evaluate(model, dataloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    return test_loss, accuracy

if __name__ == "__main__":
    # Instantiate model
    # model = CIF100Net().to(device)
    model = Net2().to(device)
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 150
    # for epoch in range(num_epochs):
    #     train_loss, train_accuracy = train(model, train_loader_mnist, criterion, optimizer)
    #     test_loss, test_accuracy = evaluate(model, test_loader_mnist, criterion)
    #     print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # for epoch in range(num_epochs):
    #     train_loss, train_accuracy = train(model, train_loader_fashion, criterion, optimizer)
    #     test_loss, test_accuracy = evaluate(model, test_loader_fashion, criterion)
    #     print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    #
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader_CIFAR, criterion, optimizer)
        test_loss, test_accuracy = evaluate(model, test_loader_CIFAR, criterion)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")