from torch import nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, kernel_size=5, padding=2, out_channels=6), nn.ReLU())
        self.sub2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=6, kernel_size=5, out_channels=16), nn.ReLU())
        self.sub4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=16, kernel_size=5, out_channels=120), nn.ReLU())
        self.full6 = nn.Sequential(nn.Flatten(), nn.Linear(in_features=120, out_features=84), nn.ReLU())
        self.full7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sub2(x)
        x = self.conv3(x)
        x = self.sub4(x)
        x = self.conv5(x)
        x = self.full6(x)
        x = self.full7(x)
        return x

class FberNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, kernel_size=5, padding=2, out_channels=3), nn.ReLU())
        self.sub2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=3, kernel_size=5, out_channels=10), nn.ReLU())
        self.sub4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=10, kernel_size=5, out_channels=30), nn.ReLU())
        self.full6 = nn.Sequential(nn.Flatten(), nn.Linear(in_features=30, out_features=10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.sub2(x)
        x = self.conv3(x)
        x = self.sub4(x)
        x = self.conv5(x)
        x = self.full6(x)
        return x