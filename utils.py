import random
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import json

from collections import OrderedDict
from typing import Any, Tuple

def save_args(path, epoch, iteration, ModelArguments, DataArguments):
    data_path = path+'.json'
    args = {}
    args['ModelArguments'] = vars(ModelArguments)
    args['DataArguments'] = vars(DataArguments)
    args['best_epoch'] = epoch
    args['best_iteration'] = iteration
    with open(data_path, 'w') as f:
        json.dump(args, f)
        
def save_loss(path='checkpoint_loss', loss_data=[]):
    data_path = path+'.json'
    args = {}
    args['loss_data'] = loss_data
    with open(data_path, 'w') as f:
        json.dump(args, f)

class ModelOutput(OrderedDict):
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)
        
    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] for k in self.keys())

class Output(ModelOutput):
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None


class ContrastiveDataset(Dataset):
    def __init__(self, tokenizer, df, img_path = None, img_id = None,org_col=None, neg_col=None):
        super().__init__()
        self.context_length = 77
        self.df = df
        self.tokenizer = tokenizer
        self.orgs = []
        self.negs = [] 
        self.img_path = img_path
        self.img_id = img_id
        self.org_col = org_col
        self.neg_col = neg_col
        
        for idx in tqdm(range(len(df))):
            row = self.df.iloc[idx] 
            
            o_row = row[self.org_col]
            org_input = self.tokenizer(text=o_row, return_tensors="pt", padding="max_length",
                                       max_length=self.context_length, truncation=True)
            
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])
            self.orgs.append(org_input)

            if self.neg_col != None:
                n_row = row[self.neg_col]
                neg_input = self.tokenizer(text=n_row, return_tensors="pt", padding="max_length",
                                            max_length=self.context_length, truncation=True)
            
                neg_input['input_ids'] = torch.squeeze(neg_input['input_ids'])
                neg_input['attention_mask'] = torch.squeeze(neg_input['attention_mask'])
                self.negs.append(neg_input)

    def __len__(self):
        return len(self.orgs)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_input = torch.load(f"{self.img_path}/{row[self.img_id]}.pt")
            
        if self.neg_col == None:
            return self.orgs[idx], img_input
        else:
            return self.orgs[idx], self.negs[idx], img_input

def create_data_loader(tokenizer=None, df=None, batch_size=None, num_workers=None, shuffle = None, 
                       drop_last=None, org_col=None, neg_col=None,  pin_memory=False, image_tensor=None, image_id=None):
    cd = ContrastiveDataset(tokenizer, df, image_tensor, image_id, org_col, neg_col)

    return DataLoader(
        cd,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory
    )

def set_seed(seed):
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def contrastive_tuplify_with_device(batch, device):
    return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                    batch[0]['attention_mask'].to(device, dtype=torch.long),
                    batch[1]['input_ids'].to(device, dtype=torch.long),
                    batch[1]['attention_mask'].to(device, dtype=torch.long),
                    batch[2].to(device, dtype=torch.float)])
    
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
        self.best_epoch = None
        self.best_iteration = None
    def __call__(self, val_loss, model, epoch, iteration):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
            self.best_iteration = iteration
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
            self.best_iteration = iteration
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
