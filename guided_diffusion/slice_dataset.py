import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

data_paths = ['/srv/tmp/brandon/pm_dataset_planes_3/pm_%04d.npy'%i for i in range(2500)]

def get_loader_img_cond(data_paths, batch_size, normalize=False): 

    dataset = PlaneDatasetConditional(data_paths, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, dataloader

class PlaneDatasetConditional(Dataset):

    def __init__(self, data_paths, normalize=True):
        self.data_paths = data_paths
        self.normalize = normalize 
        self.num_planes = np.load(self.data_paths[0]).shape[0]

    def normalize_fn(self, img): 
        return img / 10
    def __len__(self):
        return len(self.data_paths) * self.num_planes
    
    def __getitem__(self, idx):
        idx_cube = idx // self.num_planes 
        idx_plane = idx % self.num_planes
        if self.normalize: 
            data = self.normalize_fn(torch.tensor(np.load(self.data_paths[idx_cube])[idx_plane][None, ...]))
        else: 
            data = torch.tensor(np.load(self.data_paths[idx_cube])[idx_plane][None, ...])
        out_dict = {"y": np.array(idx_plane, dtype=np.int64)}
        return data.float(), out_dict

def load_data(
    *,
    data_paths,
    batch_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    normalize=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    dataset, loader = get_loader_img_cond(data_paths, batch_size, normalize=normalize)
    
    while True:
        yield from loader