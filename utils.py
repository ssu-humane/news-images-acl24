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
    def __init__(self, processor, df):
        super().__init__()
        self.context_length = 77
        self.df = df
        self.processor = processor
        self.org = []

        for idx in tqdm(range(len(df))):
            org = df.iloc[idx]

            org_input = self.processor(text=org.title, return_tensors="pt", padding="max_length",
                                       max_length=self.context_length)
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])

            self.org.append(org_input)

    def __len__(self):
        return len(self.org)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{DataArguments.IMAGE_PATH}/{row['link'][-19:]}.jpg")
        img_input = self.processor(images=image, return_tensors="pt", padding="max_length",
                                   max_length=self.context_length)
        img_input['pixel_values'] = torch.squeeze(img_input['pixel_values'])
        label = torch.from_numpy(np.asarray(row['label']))
        return self.org[idx], img_input, label

# 모든 loss에 적용할 수 있도록 코드 수정 필요
class ContrastiveDataset(Dataset):
    def __init__(self, processor, df):
        super().__init__()
        self.context_length = 77
        self.df = df
        self.processor = processor
        self.orgs = []
        self.negs = []

        for idx in tqdm(range(len(df))):
            row = self.df.iloc[idx] # dataFrame에 모두 넣어 두고 분할 할 것인지 각각 분할할 것인지 결정

            org_input = self.processor(text=row.title, return_tensors="pt", padding="max_length",
                                       max_length=self.context_length) # CLIP 허용 length 77로 모두 padding
            org_input['input_ids'] = torch.squeeze(org_input['input_ids']) # processor가 붙여주는 배치 제거
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])
            
            if DataArguments.NEGATIVE_TITLE == 'SHUFFLE':
                neg_input = self.processor(text=row.HN_SHUF, return_tensors="pt", padding="max_length",
                                           max_length=self.context_length)
            elif DataArguments.NEGATIVE_TITLE == 'MASK_PREDICT':
                neg_input = self.processor(text=row.HN_MASK_PREDICT, return_tensors="pt", padding="max_length",
                                           max_length=self.context_length)
            else:
                print("Negative Data Load error")
                
            neg_input['input_ids'] = torch.squeeze(neg_input['input_ids'])
            neg_input['attention_mask'] = torch.squeeze(neg_input['attention_mask'])
                        
            self.orgs.append(org_input)
            self.negs.append(neg_input)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image = Image.open(f"{DataArguments.IMAGE_PATH}/{row['source']}/{row['hash_id']}.png").convert('RGB') # 메모리 load 너무 커서 나눔
        img_input = self.processor(images=image, return_tensors="pt", padding="max_length",
                                    max_length=self.context_length)
        img_input['pixel_values'] = torch.squeeze(img_input['pixel_values'])
            
        return self.orgs[idx], self.negs[idx], img_input

def create_data_loader(processor=None, df=None, batch_size=None, num_workers=None, shuffle = None, drop_last=None):
    if ModelArguments.TRAIN_MODEL == 'contrastive':
        cd = ContrastiveDataset(processor, df)
    else:
        cd = ClassifierDataset(processor, df)

    return DataLoader(
        cd,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last
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
                      batch[1]['pixel_values'].to(device, dtype=torch.float),
                      batch[2].to(device, dtype=torch.float)])

def contrastive_tuplify_with_device(batch, device):
    if ModelArguments.Contrastive_Mode == 'imgtotxt':
        return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                      batch[0]['attention_mask'].to(device, dtype=torch.long),
                      batch[2]['pixel_values'].to(device, dtype=torch.float)])
    elif ModelArguments.Contrastive_Mode == 'imgtotxt_H':
        return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                      batch[0]['attention_mask'].to(device, dtype=torch.long),
                      batch[1]['input_ids'].to(device, dtype=torch.long),
                      batch[1]['attention_mask'].to(device, dtype=torch.long),
                      batch[2]['pixel_values'].to(device, dtype=torch.float)])
    elif ModelArguments.Contrastive_Mode == 'txttotxt_H':
        return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                      batch[0]['attention_mask'].to(device, dtype=torch.long),
                      batch[1]['input_ids'].to(device, dtype=torch.long),
                      batch[1]['attention_mask'].to(device, dtype=torch.long)])
    elif ModelArguments.Contrastive_Mode =='imgtoimg_H_txttotxt_H':
        return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                      batch[0]['attention_mask'].to(device, dtype=torch.long),
                      batch[1]['input_ids'].to(device, dtype=torch.long),
                      batch[1]['attention_mask'].to(device, dtype=torch.long),
                      batch[2]['pixel_values'].to(device, dtype=torch.float)])
    else:
        return -1