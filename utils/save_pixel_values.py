import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor
import argparse

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def main(args):
    df = pd.read_pickle(args.data_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image = Image.open(f'{args.image_path}/{row[args.image_id]}')
        pixel = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        torch.save(pixel, f'{args.save_path}/{row[args.image_id]}.pt')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.pkl')
    parser.add_argument('--image_id', type=str, default='image_id')
    parser.add_argument('--save_path', type=str, default='./pixel_values') 
    parser.add_argument('--image_path', type=str, default='./imagefolder') 
    args = parser.parse_args()
    main(args)
