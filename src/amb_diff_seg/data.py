from pathlib import Path

import typer
from torch.utils.data import Dataset
import torch
import pickle
import numpy as np
import os
import albumentations as A
import copy

aug_train = A.Compose([A.D4(),
                       A.RandomGamma(p=0.3),
                       A.RandomBrightnessContrast(p=0.3),
                       A.ShiftScaleRotate(rotate_limit=20,p=0.3,border_mode=0),
                       A.Normalize(always_apply=True, p=1)])

aug_eval = A.Compose([A.Normalize(always_apply=True, p=1)])

aug_crop64 = A.RandomCrop(width=64, height=64, always_apply=True, p=1) 

class LidcDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_folder : str = "./data", 
                split : str = "train",
                random_crop64 : bool = False,
                training : bool = True 
                ) -> None:
        self.training = training
        self.random_crop64 = random_crop64
        self.data_folder = data_folder
        data = pickle.load(open(os.path.join(self.data_folder,"data_lidc.pickle"),"rb"))
        patient_ids = [k.split("_")[0] for k in data.keys()]
        test_ids = np.loadtxt(os.path.join(self.data_folder,"test_ids.txt"), dtype=str)
        vali_ids = np.loadtxt(os.path.join(self.data_folder,"vali_ids.txt"), dtype=str)
        train_ids = np.setdiff1d(patient_ids, np.concatenate([test_ids, vali_ids]))

        use_ids = {"train": train_ids, "vali": vali_ids, "test": test_ids}

        self.items = []
        self.items_ids = []
        for k,v in data.items():
            patient_id = k.split("_")[0]
            if patient_id in use_ids[split]:
                self.items.append({"image": np.clip(v["image"]*255,0,255).astype(np.uint8),
                                   "mask": np.stack(v["masks"], axis=2)})
                self.items_ids.append(patient_id)
        del data


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.items)


    def __getitem__(self, idx: int):
        """Return a given sample from the dataset."""
        item = copy.deepcopy(self.items[idx])
        
        item["image"] = item["image"][:,:,None].repeat(3, axis=2)
        if self.training:
            item = aug_train(**item)
        else:
            item = aug_eval(**item)
        if self.random_crop64:
            item = aug_crop64(**item)
        image = torch.from_numpy(item["image"]).permute(2,0,1).float()

        mask = torch.from_numpy(item["mask"]).permute(2,0,1).float()
        return image, mask

def get_data(random_crop64=False,
            data_folder="./data",
            training=True,
            batch_size=16,
            return_type="dli"):
    assert return_type in ["dli","ds","dl"], "return_type must be one of ['dli','ds','dl']"
    rc64 = random_crop64
    bs = batch_size
    kw = {"data_folder": data_folder, "random_crop64": rc64, "training": training}
    train_dataset = LidcDataset(split="train", **kw)
    vali_dataset = LidcDataset(split="vali", **kw)
    test_dataset = LidcDataset(split="test", **kw)

    if return_type=="ds":
        return train_dataset, vali_dataset, test_dataset
    
    kw ={"batch_size": bs, "drop_last": training, "num_workers": 0, "shuffle": training}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **kw)
    vali_dataloader = torch.utils.data.DataLoader(vali_dataset, **kw)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,  **kw)

    if return_type == "dl":
        return train_dataloader, vali_dataloader, test_dataloader 
    if return_type == "dli":
        return DataloaderIterator(train_dataloader), DataloaderIterator(vali_dataloader), DataloaderIterator(test_dataloader)

class DataloaderIterator():
    """
    Class which takes a pytorch dataloader and enables next() ad infinum and 
    self.partial_epoch gives an iterator which only iterates on a ratio of 
    an epoch 
    """
    def __init__(self,dataloader):
        """ initialize the dataloader wrapper
        Args:
            dataloader (torch.utils.data.dataloader.DataLoader): dataloader to sample from
        """
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)
        self.partial_flag = False
        self.partial_round_err = 0

    def __len__(self):
        return len(self.dataloader)
    
    def reset(self):
        """reset the dataloader iterator"""
        self.iter = iter(self.dataloader)
        self.partial_flag = False
        self.partial_round_err = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.partial_flag:
            if self.partial_counter==self.partial_counter_end:
                self.partial_flag = False
                raise StopIteration
        try:
            batch = next(self.iter)
        except StopIteration:    
            self.iter = iter(self.dataloader)
            batch = next(self.iter)
        if self.partial_flag:
            self.partial_counter += 1
        return batch

    def partial_epoch(self,ratio):
        """returns an iterable stopping after a partial epoch 
        Args:
            ratio (float): positive float denoting the ratio of the epoch.
                           e.g. 0.2 will give 20% of an epoch, 1.5 will
                           give one and a half epoch
        Returns:
            iterable: partial epoch iterable
        """
        self.partial_flag = True
        self.partial_counter_end_unrounded = len(self.dataloader)*ratio+self.partial_round_err
        self.partial_counter_end = int(round(self.partial_counter_end_unrounded))
        self.partial_round_err = self.partial_counter_end_unrounded-self.partial_counter_end
        self.partial_counter = 0
        if self.partial_counter_end==0:
            self.partial_counter_end = 1
        return iter(self)

if __name__ == "__main__":
    train_dli, vali_dli, test_dli = get_data(random_crop64=False,
                                            data_folder="./data",
                                            training=True,
                                            batch_size=16,
                                            return_type="dli")
    batch = next(train_dli)
    print(batch[0].shape)
    print(batch[1].shape)
    print("done")