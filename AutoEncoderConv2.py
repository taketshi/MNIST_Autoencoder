import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time

class AutoEncoderConv2(nn.Module):
    def __init__(self):
        super(AutoEncoderConv2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d( in_channels = 1, out_channels = 4, kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.MaxPool2d(kernel_size = 2, stride=2),)
        
        self.fc1 = nn.Linear(4 * 7 * 7, 50)
        self.fc2 = nn.Linear(50, 4 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 4, out_channels = 4, kernel_size = 2, stride = 2),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.ConvTranspose2d(in_channels = 4, out_channels = 1, kernel_size = 2, stride = 2),
            nn.Sigmoid())
    
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0),-1)
        encoded = self.fc1(encoded)

        # Decoder
        decoded = self.fc2(encoded)
        decoded = decoded.view(decoded.size(0), 4, 7, 7)
        decoded = self.decoder(decoded)

        return decoded
    
    def train(self, dataset, criterion = 'mse', optimizer = 'adam', lr = 0.01, batch_size = 1, epochs = 5, verbose = 5):
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Criterion
        if criterion == 'mse':
            criterion = nn.MSELoss()
        elif criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()

        # Optimizer
        if optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr = lr)
        elif optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr = lr)

        losses = []
        start_time = time.time()
        for i in range(epochs):
            batch_n = 0
            for [data] in data_loader:

                # Reset Gradients
                optimizer.zero_grad()

                # Forward Pass
                pred =  self.forward(data)

                # Loss Function
                loss = criterion(pred, data)
                loss.backward()

                # Upgrade Weights and Biases
                optimizer.step()

                batch_n += 1
                if verbose != 0 and batch_n % verbose == 0:
                    print('Epoch:', i + 1, '| Batch number:', batch_n , 
                          '| Loss:', loss.item(), '| Time:',   round(time.time() - start_time, 2), 's')
                    start_time = time.time()


            losses.append(loss.item())

        return losses