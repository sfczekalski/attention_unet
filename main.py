from utils import *

data_dir = 'data'
batch_size = 4
dataloaders = get_dataloader_sep_folder(data_dir, batch_size=batch_size)
