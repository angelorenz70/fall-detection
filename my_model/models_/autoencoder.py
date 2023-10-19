import torch
import torch.nn as nn

class LeftArmAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 6),  
            torch.nn.ReLU(),
            torch.nn.Linear(6, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, input_size),  
        )
        # Set the data type for model parameters
        self.to(torch.float32)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class RightArmAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 6),  
            torch.nn.ReLU(),
            torch.nn.Linear(6, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, input_size),  
        )
        # Set the data type for model parameters
        self.to(torch.float32)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class TrunkAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 8), 
            torch.nn.ReLU(),
            # torch.nn.Linear(8, 16),
            # torch.nn.ReLU(),
            # torch.nn.Linear(16, 8),
            # torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
            
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            # torch.nn.Linear(8, 16),
            # torch.nn.ReLU(),
            # torch.nn.Linear(16, 32),
            # torch.nn.ReLU(),
            torch.nn.Linear(8, input_size),  
        )
        # Set the data type for model parameters
        self.to(torch.float32)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LeftLegAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 6),  
            torch.nn.ReLU(),
            torch.nn.Linear(6, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, input_size),  
        )
        # Set the data type for model parameters
        self.to(torch.float32)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class RightLegAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 6),  
            torch.nn.ReLU(),
            torch.nn.Linear(6, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, input_size),  
        )
        # Set the data type for model parameters
        self.to(torch.float32)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

