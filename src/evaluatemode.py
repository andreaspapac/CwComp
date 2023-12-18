from Datasets import *
from torch.utils.data import DataLoader
from utils import *
import numpy as np
from torch.autograd import Variable
from Models import *
import os

CUDA_LAUNCH_BLOCKING = 1.
from torchinfo import summary

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.initial_seed(),
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

n_epochs_d = 50

sf_train_losses = []
sf_test_losses = []
gd_train_losses = []
gd_test_losses = []

layerwise_loss = []
batch_size = 128

n_repeats = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # int(round(60000/batch_size))+1
show_iters = 200
lr_decay = False

ifmnist = False
model_id = 'cifar_cfse_nocas_l2'
data_path = 'C:/Users/Andreas/Desktop/PhD/NeurIPS_Re/src/data/'
train_data = CIFAR10(data_path, train=True, download=True)
test_data = CIFAR10(data_path, train=False, download=True)


# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 80, kernel_size=3, stride=1, padding=1, groups=10)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 240, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(240)
        self.conv4 = nn.Conv2d(240, 480, kernel_size=3, stride=1, padding=1, groups=10)
        self.bn4 = nn.BatchNorm2d(480)
        self.fc1 = nn.Linear(30720, 10)
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

if __name__ == "__main__":
    # torch.manual_seed(1234)
    seed_everything(seed=52)

    # Instantiate model
    model = Net()

    summary(model, (1, 3, 32, 32))

