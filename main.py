from utils import *
from model import AttentionUNet
from torchvision.utils import make_grid
from utils import plot_batch
from train import train_model
from loss import dice_loss
import torch.nn as nn


def train():
    model = AttentionUNet()
    model.train()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trained_model = train_model(model, dataloaders, optimizer, bpath=bpath, num_epochs=epochs)

    torch.save(trained_model, os.path.join(bpath, 'weights.pt'))


bpath = '.'
data_dir = 'data'
batch_size = 4
epochs = 1
dataloaders = get_data_loaders(data_dir, batch_size=batch_size)

plot_batch(dataloaders, batch_size)
