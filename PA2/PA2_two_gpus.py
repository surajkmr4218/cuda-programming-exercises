import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pytorch_lightning as pl

# Load the CIFAR-10 training dataset
training_data = datasets.CIFAR10(
    root="data", train=True, download=True, transform=transforms.ToTensor()
)

# The cnn model class is now a LightningModule instead of nn.Module.
class EX_CNN(pl.LightningModule):
    """
    Example CNN model for image classification.
    """
    def __init__(self):
        super(EX_CNN, self).__init__()
        # Define convolutional layers
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
        )
        # Define fully connected layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=10),  # Output raw logits instead of sigmoid since we are doing CrossEntropyLoss
        )
        # Specify the loss function here so training_step can use it.
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.convolution_layers(x)  # Apply convolutional layers
        x = torch.flatten(x, start_dim=1)  # Flatten the output for fully connected layers
        x = self.linear_layers(x)  # Apply fully connected layers
        return x

    def training_step(self, batch, batch_idx):
        """
        Separate training_step method so lightning can backpropagate
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Returns original Adam optimizer for lightning
        """
        return optim.Adam(params=self.parameters(), lr=0.001)

if __name__ == "__main__":
    import time

    results = {}
    for bs in [32, 64, 128, 256]:
        # New loader and fresh model for each batch size
        loader = DataLoader(training_data, batch_size=bs, shuffle=True)
        Ex_Net = EX_CNN()

        # Train on 2 GPUs for 2 epochs (DDP = one process per GPU)
        trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp",
                             max_epochs=2, log_every_n_steps=20,
                             enable_checkpointing=False)

        start = time.time()
        trainer.fit(Ex_Net, train_dataloaders=loader)
        results[bs] = time.time() - start
        if trainer.global_rank == 0:
            print(f"Batch size {bs}: {results[bs]:.1f}s")

    if trainer.global_rank == 0:
        print("\nSummary (2 epochs, 2 GPUs):")
        for bs, t in results.items():
            print(f"  bs={bs:4d}: {t:.1f}s")
        print("Finished training.")