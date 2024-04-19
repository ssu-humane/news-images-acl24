import argparse
import pandas as pd
import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_web_trf')

ntt = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
def mask_ntt(text):
    doc = nlp(text)
    mask_list = []
    for ent in doc.ents:
        if ent.label_ in ntt:
            mask_list.append((ent.text, ent.label_,(ent.start_char, ent.end_char)))
    return mask_list

def main(args):
    df = pd.read_pickle(args.data_path)

    global_ntts = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row[args.target_text]
        ntt = mask_ntt(text)
        global_ntts.append(ntt)
    df['NTT'] = global_ntts

    df.to_pickle(args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train.pkl')
    parser.add_argument('--save_path', type=str, default='./train_ntt.pkl')
    parser.add_argument('--target_text', type=str, default='summary')

    args = parser.parse_args()
    main(args)