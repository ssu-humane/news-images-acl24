import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


from transformers import AutoTokenizer

from models import ContrastiveBaseLineModel
from utils import (
    create_data_loader,
    set_seed,
    AverageMeter,
    contrastive_tuplify_with_device,
    EarlyStopping,
    save_args,
    save_loss
)
from config import ModelArguments, DataArguments
from contrastiveLoss import Contrastive_Loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(ModelArguments, DataArguments):
    set_seed(ModelArguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(ModelArguments.CLIP_MODEL)
    
    encoder = ContrastiveBaseLineModel(ModelArguments)
    encoder.vision_freeze()
    encoder.text_freeze()

    if ModelArguments.DATAPARALLEL:
        encoder = nn.DataParallel(encoder)
    encoder = encoder.to(device)

    loss_func = Contrastive_Loss(temperature=ModelArguments.LOSS_TEMPERATURE,
                                    batch_size=ModelArguments.batch_size)
    
    optimizer = optim.AdamW(encoder.parameters(), lr=ModelArguments.learning_rate)

    train_df = pd.read_pickle(DataArguments.TRAIN_DATA_PATH)
    valid_df = pd.read_pickle(DataArguments.VALID_DATA_PATH)
    
    
    train_data_loader = create_data_loader(tokenizer = tokenizer, df = train_df, batch_size = ModelArguments.batch_size,
                                            num_workers=ModelArguments.num_workers, shuffle=DataArguments.TRAIN_DATA_LOADER_SHUFFLE,
                                            drop_last=DataArguments.TRAIN_DATA_LOADER_DROP_LAST, org_col=DataArguments.ORG_TITLE,
                                            neg_col=DataArguments.NEGATIVE_TITLE, pin_memory=DataArguments.pin_memory,
                                            image_tensor=DataArguments.IMAGE_TENSOR, image_id=DataArguments.IMAGE_ID)
    valid_data_loader = create_data_loader(tokenizer = tokenizer, df = valid_df, batch_size = ModelArguments.batch_size,
                                            num_workers=ModelArguments.num_workers, shuffle=DataArguments.VALID_DATA_LOADER_SHUFFLE,
                                            drop_last=DataArguments.VALID_DATA_LOADER_DROP_LAST,org_col=DataArguments.ORG_TITLE,
                                            neg_col=DataArguments.NEGATIVE_TITLE, pin_memory=DataArguments.pin_memory,
                                            image_tensor=DataArguments.IMAGE_TENSOR, image_id=DataArguments.IMAGE_ID)
    
    if ModelArguments.SCHEDULER_USE:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ModelArguments.LR_SCHEDULER_T_MAX, eta_min=0)

    early_stopping = EarlyStopping(patience=ModelArguments.patience, verbose=True)
    save_args(f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_config', early_stopping.best_epoch, early_stopping.best_iteration, ModelArguments, DataArguments)
        
    d_iter = 0
    
    loss_data = []
    for e_idx in range(1, ModelArguments.epochs + 1):
        losses = AverageMeter()

        valid_loss = []

        tbar = tqdm(train_data_loader)
        
        encoder.train()
        for _, batch in enumerate(tbar):
            if early_stopping.early_stop:
               break
           
            d_iter += ModelArguments.batch_size
            
            batch = contrastive_tuplify_with_device(batch, device)

            org_input_ids, org_attention_mask,\
            neg_input_ids, neg_attention_mask, org_pixel_values = batch
            outputs = encoder(input_ids=torch.cat([org_input_ids, neg_input_ids]),
                                attention_mask=torch.cat([org_attention_mask, neg_attention_mask]),
                                pixel_values=org_pixel_values)
            
            loss = loss_func(outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), ModelArguments.batch_size)
            tbar.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)       

            if d_iter % 20480 == 0 or d_iter == ((len(train_data_loader.dataset)//ModelArguments.batch_size)*ModelArguments.batch_size*ModelArguments.epochs):
                print('d_iter: ',d_iter)
                tbar2 = tqdm(valid_data_loader)
                encoder.eval()
                with torch.no_grad():
                    for _, batch in enumerate(tbar2):
                        batch = contrastive_tuplify_with_device(batch, device)

                        org_input_ids, org_attention_mask,\
                        neg_input_ids, neg_attention_mask, org_pixel_values = batch
                        outputs = encoder(input_ids=torch.cat([org_input_ids, neg_input_ids]),
                                        attention_mask=torch.cat([org_attention_mask, neg_attention_mask]),
                                        pixel_values=org_pixel_values)
                            
                        loss = loss_func(outputs)
                        valid_loss.append(loss.item())
                    
                    valid_loss_avg = sum(valid_loss) / len(valid_loss)
                    
                    loss_data.append([e_idx, losses.avg, 'Train', d_iter])
                    loss_data.append([e_idx, valid_loss_avg, 'Valid', d_iter])
                
                    print('Valid | epoch: ', str(e_idx), ', Valid Loss Avg: ', str(valid_loss_avg), 'd_iter:', str(d_iter))
                    
                    early_stopping(valid_loss_avg, encoder, e_idx, d_iter)
                    encoder.train()
                               
                
            if ModelArguments.SCHEDULER_USE:
                scheduler.step()
        
        if early_stopping.early_stop:
           break

    encoder.load_state_dict(torch.load('checkpoint.pt'))
    torch.save(encoder.state_dict(), f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_best.pt')

    save_args(f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_config', early_stopping.best_epoch, early_stopping.best_iteration, ModelArguments, DataArguments)
    save_loss(f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_loss', loss_data)

if __name__ == "__main__":
    main(ModelArguments, DataArguments)
   
