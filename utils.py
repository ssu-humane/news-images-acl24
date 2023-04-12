import random
import numpy as np
import torch
from tqdm.auto import tqdm
from config import ModelArguments, DataArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassifierDataset(Dataset):
    def __init__(self, processor, df, img_path = None, org_col=None):
        super().__init__()
        self.context_length = 77
        self.df = df
        self.processor = processor
        self.orgs = []
        self.img_path = img_path
        self.org_col = org_col

        for idx in tqdm(range(len(df))):
            org = self.df.iloc[idx]

            org_input = self.processor(text=org[self.org_col], return_tensors="pt", padding="max_length",
                                       max_length=self.context_length)
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])

            self.orgs.append(org_input)

    def __len__(self):
        return len(self.orgs)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_input = torch.load(f"{self.img_path}/{row.hash_id}.pt")
        label = torch.from_numpy(np.asarray(row['label']))
        return self.orgs[idx], img_input, label

# 모든 loss에 적용할 수 있도록 코드 수정 필요
class ContrastiveDataset(Dataset):
    def __init__(self, processor, df, img_path = None, org_col=None, neg_col=None):
        super().__init__()
        self.context_length = 77
        self.df = df
        self.processor = processor
        self.orgs = []
        self.negs = [] 
        self.img_path = img_path
        self.org_col = org_col
        self.neg_col = neg_col
        
        for idx in tqdm(range(len(df))):
            row = self.df.iloc[idx] # dataFrame에 모두 넣어 두고 분할 할 것인지 각각 분할할 것인지 결정
            o_row = row[self.org_col]
            n_row = row[self.neg_col]
            
            org_input = self.processor(text=o_row, return_tensors="pt", padding="max_length",
                                       max_length=self.context_length)
            
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])
            
            
            neg_input = self.processor(text=n_row, return_tensors="pt", padding="max_length",
                                       max_length=self.context_length)
            
            neg_input['input_ids'] = torch.squeeze(neg_input['input_ids'])
            neg_input['attention_mask'] = torch.squeeze(neg_input['attention_mask'])
            
            self.orgs.append(org_input)
            self.negs.append(neg_input)

    def __len__(self):
        return len(self.orgs)

    def __getitem__(self, idx):
        row = self.df.iat[idx,0]
        img_input = torch.load(f"{self.img_path}/{row}.pt")
        return self.orgs[idx], self.negs[idx], img_input


    
def create_data_loader(processor=None, df=None, batch_size=None, num_workers=None, shuffle = None, drop_last=None, org_col=None, neg_col=None,  pin_memory=False):
    if ModelArguments.TRAIN_MODEL == 'contrastive': 
        cd = ContrastiveDataset(processor, df, DataArguments.IMAGE_TENSOR, org_col, neg_col)
    else:
        cd = ClassifierDataset(processor, df, DataArguments.IMAGE_TENSOR, org_col)

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

def compute_accuracy(y_target, y_pred):
    y_pred_tag = y_pred.round()
    def_acc = accuracy_score(y_target, y_pred_tag)
    def_f1 = f1_score(y_target, y_pred_tag, average='binary')
    def_roc = roc_auc_score(y_target, y_pred)
    return def_acc, def_f1, def_roc


def classifier_tuplify_with_device(batch, device):
    return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                      batch[0]['attention_mask'].to(device, dtype=torch.long),
                      batch[1].to(device, dtype=torch.float),
                      batch[2].to(device, dtype=torch.float)])

def contrastive_tuplify_with_device(batch, device):
    if ModelArguments.Contrastive_Mode == 'imgtotxt':
        return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                      batch[0]['attention_mask'].to(device, dtype=torch.long),
                      batch[2].to(device, dtype=torch.float)])
    elif ModelArguments.Contrastive_Mode == 'imgtotxt_H':
        return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                      batch[0]['attention_mask'].to(device, dtype=torch.long),
                      batch[1]['input_ids'].to(device, dtype=torch.long),
                      batch[1]['attention_mask'].to(device, dtype=torch.long),
                      batch[2].to(device, dtype=torch.float)])
    elif ModelArguments.Contrastive_Mode == 'txttotxt_H':
        return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                      batch[0]['attention_mask'].to(device, dtype=torch.long),
                      batch[1]['input_ids'].to(device, dtype=torch.long),
                      batch[1]['attention_mask'].to(device, dtype=torch.long),
                      batch[2].to(device, dtype=torch.float)])
    elif ModelArguments.Contrastive_Mode =='imgtotxt_txttotxt_H':
        return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                      batch[0]['attention_mask'].to(device, dtype=torch.long),
                      batch[1]['input_ids'].to(device, dtype=torch.long),
                      batch[1]['attention_mask'].to(device, dtype=torch.long),
                      batch[2].to(device, dtype=torch.float)])
    else:
        return -1