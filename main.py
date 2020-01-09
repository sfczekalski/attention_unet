from utils import *
from model import AttentionUNet
from torchvision.utils import make_grid
from utils import imshow

data_dir = 'data'
batch_size = 4
dataloaders = get_data_loader(data_dir, batch_size=batch_size)

'''for sample in iter(dataloaders['training']):
    print(sample['image'])'''

dataiter = iter(dataloaders['training']).__next__()
print(dataiter['image'])


