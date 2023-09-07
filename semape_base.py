import os
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeo.datasets
from torch.utils import data
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from torchvision.models.segmentation import DeepLabV3

from ENet import ENet
from SegNet import SegNet
from segacc import SegmentationMetrics

import glob
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

class LoveDA():
    """LoveDA dataset.

    The `LoveDA <https://github.com/Junjue-Wang/LoveDA>`__ datataset is a
    semantic segmentation dataset.

    Dataset features:

    * 2713 urban scene and 3274 rural scene HSR images, spatial resolution of 0.3m
    * image source is Google Earth platform
    * total of 166768 annotated objects from Nanjing, Changzhou and Wuhan cities
    * dataset comes with predefined train, validation, and test set
    * dataset differentiates between 'rural' and 'urban' images

    Dataset format:

    * images are three-channel pngs with dimension 1024x1024
    * segmentation masks are single-channel pngs

    Dataset classes:

    1. background
    2. building
    3. road
    4. water
    5. barren
    6. forest
    7. agriculture

    No-data regions assigned with 0 and should be ignored.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2110.08733

    .. versionadded:: 0.2
    """

    scenes = ["urban", "rural"]
    splits = ["train", "val", "test"]

    info_dict = {
        "train": {
            "url": "https://zenodo.org/record/5706578/files/Train.zip?download=1",
            "filename": "Train.zip",
            "md5": "de2b196043ed9b4af1690b3f9a7d558f",
        },
        "val": {
            "url": "https://zenodo.org/record/5706578/files/Val.zip?download=1",
            "filename": "Val.zip",
            "md5": "84cae2577468ff0b5386758bb386d31d",
        },
        "test": {
            "url": "https://zenodo.org/record/5706578/files/Test.zip?download=1",
            "filename": "Test.zip",
            "md5": "a489be0090465e01fb067795d24e6b47",
        },
    }

    classes = [
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agriculture",
        "no-data",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        scene: list[str] = ["urban", "rural"],
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LoveDA dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            scene: specify whether to load only 'urban', only 'rural' or both
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            AssertionError: if ``scene`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in self.splits
        assert set(scene).intersection(
            set(self.scenes)
        ), "The possible scenes are 'rural' and/or 'urban'"
        assert len(scene) <= 2, "There are no other scenes than 'rural' or 'urban'"

        self.root = root
        self.split = split
        self.scene = scene
        self.transforms = transforms
        self.checksum = checksum

        self.url = self.info_dict[self.split]["url"]
        self.filename = self.info_dict[self.split]["filename"]
        self.md5 = self.info_dict[self.split]["md5"]

        self.directory = os.path.join(self.root, split.capitalize())
        self.scene_paths = [
            os.path.join(self.directory, s.capitalize()) for s in self.scene
        ]

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found at root directory or corrupted. "
                + "You can use download=True to download it"
            )

        self.files = self._load_files(self.scene_paths, self.split)

    def __getitem__(self, index: int):
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and mask at that index with image of dimension 3x1024x1024
            and mask of dimension 1024x1024
        """
        files = self.files[index]
        img = self._load_image(files["image"])

        #modified from dict to tuple for training or individual sample for testing
        if self.split != "test":
            mask = self._load_target(files["mask"])

        if self.transforms is not None:
            img = self.transforms(img)

        if self.split == 'train':
            return img, mask
        else:
            return img    

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return len(self.files)

    def _load_files(self, scene_paths: list[str], split: str) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            scene_paths: contains one or two paths, depending on whether user has
                         specified only 'rural', 'only 'urban' or both
            split: subset of dataset, one of [train, val, test]
        """
        images = []

        for s in scene_paths:
            images.extend(glob.glob(os.path.join(s, "images_png", "*.png")))

        images = sorted(images)

        if self.split != "test":
            masks = [image.replace("images_png", "masks_png") for image in images]
            files = [
                dict(image=image, mask=mask) for image, mask, in zip(images, masks)
            ]
        else:
            files = [dict(image=image) for image in images]

        return files

    def _load_image(self, path: str):
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the loaded image
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array).float()
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, path: str):
        """Load a single mask corresponding to image.

        Args:
            path: path to the mask

        Returns:
            the mask of the image
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("L"))
            tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.long)
            return tensor

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """
        for s in self.scene_paths:
            if not os.path.exists(s):
                return False

        return True

    def plot(
        self, sample: dict[str, Tensor], suptitle: Optional[str] = None
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        if self.split != "test":
            image, mask = sample["image"], sample["mask"]
            ncols = 2
        else:
            image = sample["image"]
            ncols = 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))

        if self.split != "test":
            axs[0].imshow(image.permute(1, 2, 0))
            axs[0].axis("off")
            axs[1].imshow(mask)
            axs[1].axis("off")
        else:
            axs.imshow(image.permute(1, 2, 0))
            axs.axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            #t.sub_(m).div_(s) #normalize code
        return tensor

torch.cuda.set_device(0)

torch.cuda.is_available()

class semantic_dataset(data.Dataset):
    def __init__(self, split, height, width, transform = None):
        self.valid_labels = [1, 2, 3, 4, 5, 6, 7]
        self.class_names = ['other', 'wall', 'road', 'vegetation', 'vehicle', 'roof', 'water']
        self.void_labels = []
        self.ignore_index = 0
        self.class_map = dict(zip(self.valid_labels, range(8)))
        self.split = split
        if self.split == 'train':
            self.img_path = r"C:\Users\Shadow\data\Train\Rural\images_png"    
            self.mask_path = r"C:\Users\Shadow\data\Train\Rural\masks_png"
        self.height = height
        self.width = width
        self.transform = transform
        
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)
        
    def __len__(self):
        return(len(self.img_list))
    
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
       # print(img.shape)
        img = img.resize((self.width, self.height))

        mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.width, self.height))
        mask = self.encode_segmap(mask)
        assert(mask.shape == (self.height, self.width))
    
        if self.transform:
            img = self.transform(img)
            assert(img.shape == (3, self.height, self.width))
        else :
            assert(img.shape == (self.height, self.width, 3))

        return img, mask
   
    #sets void classes to zero so they won't be considered for training
    def encode_segmap(self, mask):
        for voidc in self.void_labels :
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels :
            mask[mask == validc] = self.class_map[validc]
        return mask
    
    def get_filenames(self, path):
        files_list = []
        for filename in os.listdir(path): 
            files_list.append(os.path.join(path, filename))

        return files_list  

torch.cuda.set_device(0)

torch.cuda.is_available()

def train_val_dataset(dataset, val_split=0.20):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

train_set = semantic_dataset(split = 'train', height=408, width=408, transform=Compose([ToTensor]))

train_datasets = train_val_dataset(train_set, )

#lightning module for easy network swapping
class SegModel(pl.LightningModule):
    def __init__(self):
        super(SegModel, self).__init__()
        self.batch_size = 48
        self.learning_rate = 1e-2
        #self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = 19)
        #self.net = UNet(num_classes = 19, bilinear = False)
        #self.net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, progress = True, num_classes = 7)
        self.net = SegNet(kernel_size=3)
        self.transform = transforms.Compose([
           # transforms.ColorJitter(contrast=0.1),
            transforms.ToTensor(),
           # transforms.Normalize(mean = [0.4861, 0.4869, 0.4411], std =[0.1810, 0.1684, 0.2066])
        ])
        self.geo_transform = transforms.Compose([
            transforms.ColorJitter(contrast=0.3),
            transforms.Resize(256),
            #transforms.ToTensor()
        ])
        self.trainset = train_datasets['train']
        self.testset = train_datasets['val']

    def forward(self, x):
        return self.net(x)
    
    #training step standard
    def training_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=0)
       #  loss_val = F.binary_cross_entropy(model(out), mask)
#         print(loss.shape)
        return {'loss' : loss_val}
    
    #optimizers
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]
    
    #dataloaders
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True, num_workers=8, persistent_workers=True, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size = 1, shuffle = True)

#main training loop
if __name__ == '__main__':

    model = SegModel()
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=r"C:\Users\Shadow\Downloads\VDD",
        save_last=True,
        save_weights_only=False,
        verbose = True, 
        mode = 'min',
        every_n_epochs=1
    )

    torch.cuda.set_device(0)

    torch.cuda.is_available()

    #comment these out to remove training loop
    #trainer = pl.Trainer(accelerator='gpu', max_epochs = 200, precision='16', devices=[0], callbacks = [checkpoint_callback], log_every_n_steps=5)

   # trainer.fit(model)
                                                                                                                                                            
    #loads the last trained model
    model = SegModel()
    checkpoint = torch.load(r"C:\Users\Shadow\Downloads\VDD\last-v38.ckpt", map_location = lambda storage, loc : storage)
    #checkpoint = torch.load(r"C:\Users\Shadow\Downloads\VDD\SegNet\last-v1.ckpt", map_location = lambda storage, loc : storage)

    model.load_state_dict(checkpoint['state_dict'])
    model.net.eval()

    testloader = model.test_dataloader()

    accuracy = SegmentationMetrics()

    for _ in range(5):
            
        batch = next(iter(testloader))
        img, mask = batch
        y = model.forward(img)
        mask_pred = y.cpu().detach().numpy()
        mask_pred_bw = np.argmax(mask_pred[0], axis = 0)

        # unorm = UnNormalize(mean = [0.4861, 0.4869, 0.4411], std = [0.1810, 0.1684, 0.2066])
        #img2 = unorm(img)
        img2 = img.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()

        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(img2[0])
        axes[1].imshow(mask_pred_bw)
        #plt.savefig('output.png')
        plt.show()