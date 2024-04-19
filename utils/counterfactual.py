import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel,AutoTokenizer, BertForMaskedLM
import torch
import random
import argparse
import torch
import re
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

model.to(device)
model.eval()

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip.to(device)
clip.eval()

def check_per(per):
    cnt = 0
    pers = []
    for v in per:
        n_len = len(re.sub(r'[^\w\s]', '', v[0]).split())
        if n_len > 0 and len(v[0]) > 1:
            pers.append(v)
        else: cnt +=1
    return pers, cnt

def smoothing(v, t):
    return torch.exp(v/t)/torch.sum(torch.exp(v/t))

def select_token(v):
    return torch.multinomial(v, 1)

def masked_text(text, aug_list):
    aug = sorted(aug_list,key=lambda x:x[1], reverse=True)
    for a in aug:
        text = text[:a[1][0]] + '[MASK]' + text[a[1][1]:]
    return text

def masked_predict(text, t):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index]
        for tensor in predicted_token_id:
            predict_word = tokenizer.decode(select_token(smoothing(tensor, t)))
            text = text.replace('[MASK]', predict_word, 1)
    return text

def main(args):
    random.seed(args.seed)
    
    df = pd.read_pickle(args.data_path)

    df['text_char_len'] = df.apply(lambda x:len(x[args.target_text].replace(' ', '')), axis=1)

    cnt = 0
    pnps = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        np = row.NTT
        pnp, c = check_per(np)
        cnt += c
        pnps.append(pnp)
    df['PNTT'] = pnps

    survive = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        ntt = row.PNTT
        if len(ntt) == 0:
            survive.append(None)
        else:
            sur = []
            for n in ntt:
                if n[1] == 'PERSON':
                    sur.append(n)
            if len(sur) == 0:
                survive.append(None)
            else:
                survive.append(sur)
    df['PERSON'] = survive
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    per_sets = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        per_set = []
        for n in row['PERSON']:
            n_len = len(n[0].replace(' ', ''))
            per_set.append((n[0], n[2], n_len))
        per_sets.append(per_set)
    df['per_set'] = per_sets

    max_samples = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        title_len = row.text_char_len
        div_title_len = int(title_len*0.3)
        person_set = row.per_set
        cnt = len(person_set)
        if cnt > 0:
            while div_title_len < sum([i[2] for i in person_set]):
                random_element = random.choice(person_set)
                person_set.remove(random_element)
                cnt -= 1
                if cnt == 0: break
            max_samples.append(person_set)
        else:
            max_samples.append([])
    df['max_samples'] = max_samples

    df = df[df['max_samples'].apply(lambda x: len(x) > 0)]
    df.reset_index(drop=True,inplace=True)

    masked = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row[args.target_text]
        mask_text = masked_text(text, row.max_samples)
        masked.append(mask_text)
    df['masked_text'] = masked

    cossim = torch.nn.CosineSimilarity()
    texts = []
    predict_t2_0_cossim = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        org_text = row[args.target_text].lower()
        org_similarity = row['clip-vit-large-patch14-cossim']
        image = torch.load(f'{args.pixel_values}/{row[args.image_id]}.pt').unsqueeze(0).to(device)
        text = row.masked_text
        
        for _ in range(10):
            mask_predict_text = masked_predict(text, args.temperature)
            inputs = processor(text=[mask_predict_text], return_tensors='pt').to(device)
            outputs = clip(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=image)
            similarity = cossim(outputs.text_embeds, outputs.image_embeds).item()

            if (org_similarity > similarity) and (org_text != mask_predict_text.lower()):
                break

        texts.append(mask_predict_text)
        predict_t2_0_cossim.append(similarity)
    df['predict_t2_0'] = texts
    df['predict_t2_0_cossim'] = predict_t2_0_cossim


    df['diff'] = df.apply(lambda x:x['clip-vit-large-patch14-cossim'] - x['predict_t2_0_cossim'], axis=1)
    df = df[df['diff'] > 0]
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='valid_ntt_cossim.pkl')
    parser.add_argument('--data', type=str, default='BBC')
    parser.add_argument('--pixel_values', type=str, default='./pixel_values')
    parser.add_argument('--image_id', type=str, default='image_id')
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--save_path', type=str, default='valid_ntt_cossim_CFT.pkl')
    parser.add_argument('--target_text', type=str, default='summary')
    args = parser.parse_args()
    main(args)