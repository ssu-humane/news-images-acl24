import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(args):
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    
    df = pd.read_pickle(args.data_path)
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device)
    
    cossim = torch.nn.CosineSimilarity()
    scores = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        text = processor(text=[row[args.target_text]], return_tensors='pt').to(device)
        image = torch.load(f'{args.pixel_values}/{row[args.image_id]}.pt').unsqueeze(0).to(device)
            
        with torch.no_grad():
            try:
                output = model(input_ids=text.input_ids, attention_mask= text.attention_mask, pixel_values=image)
                cossim_score = cossim(output.text_embeds, output.image_embeds).cpu().item()
                scores.append(cossim_score)
            except:
                scores.append(None)
    df['clip-vit-large-patch14-cossim'] = scores
        
    df.to_pickle(args.save_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train_ntt.pkl')
    parser.add_argument('--image_id', type=str, default='image_id')
    parser.add_argument('--pixel_values', type=str, default='./pixel_values')
    parser.add_argument('--save_path', type=str, default='./train_ntt_cossim.pkl')
    parser.add_argument('--target_text', type=str, default='summary')
    args = parser.parse_args()
    main(args)