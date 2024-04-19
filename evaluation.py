import json
import argparse
from argparse import Namespace
from collections import OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
from models import ContrastiveBaseLineModel
from utils import ContrastiveDataset
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict

def tuplify_with_device(batch, device):
    return tuple([batch[0]['input_ids'].to(device, dtype=torch.long),
                batch[0]['attention_mask'].to(device, dtype=torch.long),
                batch[1].to(device, dtype=torch.float)])

def scalings(x):
    return (x+1)/2

def threshold_score(v_pred = None, label = None, pred = None):
    v_pred = np.array(list(map(scalings, v_pred)))
    pred = np.array(list(map(scalings, pred)))
       
    v_trsd = np.median(v_pred)
    
    f_preds = np.where(pred > v_trsd, 1, 0)
    
    f1 = f1_score(label, f_preds)
    
    return f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid = pd.read_pickle(args.valid_path)
    eval_df = pd.read_pickle(args.eval_path)

    with open(f"{args.model_path}_config.json", 'r') as f:
        encoder_arguments = Namespace(**json.load(f))
        encoder_model_args = Namespace(**encoder_arguments.ModelArguments)
    model_config = encoder_model_args

    encoder = ContrastiveBaseLineModel(model_config)
    state_dict = torch.load(f'{args.model_path}_best.pt')
    try:
        encoder.load_state_dict(state_dict)
    except:
        state_dict = remove_data_parallel(state_dict)
        encoder.load_state_dict(state_dict)

    encoder.to(device)
    encoder.eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dataset = ContrastiveDataset(processor, valid, img_path=args.pixel_path, org_col=args.target_text, img_id=args.val_image_id)
    dataloader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)    

    cossim = torch.nn.CosineSimilarity()
    score = []
    tbar = tqdm(dataloader)
    with torch.no_grad():
        for _, batch in enumerate(tbar):
            batch = tuplify_with_device(batch, device)
            input_ids, attention_mask, pixel_values = batch
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            cs = cossim(outputs[0], outputs[1]).detach().cpu().numpy()
            score.append(cs)
        score = np.concatenate(score).reshape((-1,))
    valid['score'] = score
    
    score = []
    with torch.no_grad():
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
            text = row[args.eval_target_text]
            image = torch.load(f'{args.pixel_path}/{row[args.eval_image_id]}.pt').unsqueeze(0).to(device)
            inputs = processor(text=[text], return_tensors='pt', truncation=True).to(device)
            outputs = encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=image)
            cs = cossim(outputs[0], outputs[1]).detach().cpu().numpy()
            score.append(cs)
        score = np.concatenate(score).reshape((-1,))
    eval_df['score'] = score

    valid.to_pickle(f"{args.save_path}/valid_{args.model_path.split('/')[-1]}_{args.valid_path.split('/')[-1][:-4]}.pkl")
    eval_df.to_pickle(f"{args.save_path}/eval_{args.model_path.split('/')[-1]}_{args.valid_path.split('/')[-1][:-4]}.pkl")

    result = threshold_score(valid.score, label=eval_df.label, pred=eval_df.score)
    print(f"F1: {result:.3f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixel_path', type=str, default='./data/pixel_values')
    parser.add_argument('--image_path', type=str, default='./data/Imagefolder')
    parser.add_argument('--val_image_id', type=str, default='image_id')
    parser.add_argument('--eval_image_id', type=str, default='image_id')
    
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--target_text', type=str, default='summary')
    parser.add_argument('--eval_target_text', type=str, default='summary')
    
    parser.add_argument('--valid_path', type=str, default='./data/valid.pkl')
    parser.add_argument('--eval_path', type=str, default='../data/annotation.pkl')
    
    parser.add_argument('--model_path', type=str, default='../model/CFT-CLIP/CFT-CLIP_vf0_tf11')
    parser.add_argument('--save_path', type=str, default='../score/CFT-CLIP')
    
    
    args = parser.parse_args()

    main(args)