import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorchtools import EarlyStopping
from collections import OrderedDict

from transformers import CLIPProcessor
from models import ClassifierBaseLineModel, ContrastiveBaseLineModel, CLIPFineT
from utils import (
    create_data_loader,
    set_seed,
    AverageMeter,
    compute_accuracy,
    classifier_tuplify_with_device,
    contrastive_tuplify_with_device
)
from config import ModelArguments, DataArguments, save_args, save_loss
from contrastiveLoss import (
    Contrastive_Loss1,
    Contrastive_Loss2,
    Contrastive_Loss3,
    Contrastive_Loss4
)
import json
from argparse import Namespace
import time
from eval_func import evaluate_CL, evaluate_CF

def main():
    set_seed(ModelArguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ModelArguments.TRAIN_MODEL == 'contrastive':
        processor = CLIPProcessor.from_pretrained(ModelArguments.CLIP_MODEL)
        
        if ModelArguments.MLP_USE:
            print("MLP USE")
            encoder = ContrastiveBaseLineModel(ModelArguments)
        else:
            print("MLP Not USE")
            encoder = CLIPFineT(ModelArguments)
        
        if ModelArguments.TEXT_ENCODER_NO_FREEZE:
            print("image encoder freeze")
            encoder.vision_freeze()
            if ModelArguments.TEXT_ENCODER_PART_FREEZE:
                encoder.vision_all_text_part_freeze()
        else:
            print("all freeze")
            encoder.all_freeze()
    
        if ModelArguments.DATAPARALLEL:
            print("DataParallel")
            encoder = nn.DataParallel(encoder)
        encoder = encoder.to(device)

        if ModelArguments.Contrastive_Mode == 'imgtotxt':
            print('imgtotxt loss')
            loss_func = Contrastive_Loss1(temperature = ModelArguments.LOSS_TEMPERATURE,
                                          batch_size = ModelArguments.batch_size)
        elif ModelArguments.Contrastive_Mode == 'imgtotxt_H':
            print('imgtotxt_H loss')
            loss_func = Contrastive_Loss2(temperature=ModelArguments.LOSS_TEMPERATURE,
                                          batch_size=ModelArguments.batch_size)
        else:
            print("Contrastive Loss select error")
        
        if ModelArguments.ADAM:
            print("Adam")
            optimizer = optim.Adam(encoder.parameters(), lr=ModelArguments.learning_rate, betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
        else:
            print("AdamW")
            optimizer = optim.AdamW(encoder.parameters(), lr=ModelArguments.learning_rate)


    elif ModelArguments.TRAIN_MODEL == 'classifier':
        processor = CLIPProcessor.from_pretrained(ModelArguments.CLIP_MODEL)
        
        if ModelArguments.CF_ENCODER:
            print("Use Encoder")
            with open(ModelArguments.CF_ENCODER_CONFIG, 'r') as f:
                encoder_arguments = Namespace(**json.load(f))
                encoder_model_args = Namespace(**encoder_arguments.ModelArguments)
                
            if encoder_model_args.MLP_USE:
                print("MLP USE")
                encoder = ContrastiveBaseLineModel(encoder_model_args)
            else:
                print("MLP Not USE")
                encoder = CLIPFineT(encoder_model_args)
            
            print("encoder all freeze")
            encoder.all_freeze() 
            
            
            try:
                if isinstance(encoder, nn.DataParallel):  # GPU 병렬사용 적용
                    print('encoder: data parallel')
                    encoder.load_state_dict(torch.load(ModelArguments.CF_ENCODER))
                else: # GPU 병렬사용을 안할 경우
                    print('encoder: data parallel remove')
                    state_dict = torch.load(ModelArguments.CF_ENCODER) 
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`  ## module 키 제거
                        new_state_dict[name] = v
                    encoder.load_state_dict(new_state_dict)
            except:
                encoder.load_state_dict(torch.load(ModelArguments.CF_ENCODER)) # 병렬 적용 안했을 때
            encoder = encoder.to(device)
            
            classifier = ClassifierBaseLineModel(ModelArguments, encoder)
            classifier.all_freeze_encoder()
        else:
            print("Not Use Encoder")
            classifier = ClassifierBaseLineModel(ModelArguments)
            classifier.all_freeze_clip()
            
        classifier = classifier.to(device)

        loss_func = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(classifier.parameters(), lr=ModelArguments.learning_rate)

    else:
        print("Train Model select error")

    train_df = pd.read_pickle(DataArguments.TRAIN_DATA_PATH)
    valid_df = pd.read_pickle(DataArguments.VALID_DATA_PATH)
    
    
    if ModelArguments.TRAIN_MODEL == 'contrastive':
        train_data_loader = create_data_loader(processor = processor, df = train_df, batch_size = ModelArguments.batch_size,
                                               num_workers=ModelArguments.num_workers,
                                               shuffle=DataArguments.TRAIN_DATA_LOADER_SHUFFLE,
                                               drop_last=DataArguments.TRAIN_DATA_LOADER_DROP_LAST,org_col=DataArguments.ORG_TITLE,neg_col=DataArguments.NEGATIVE_TITLE, pin_memory=DataArguments.pin_memory)
        valid_data_loader = create_data_loader(processor = processor, df = valid_df, batch_size = ModelArguments.batch_size,
                                               num_workers=ModelArguments.num_workers,
                                               shuffle=DataArguments.VALID_DATA_LOADER_SHUFFLE,
                                               drop_last=DataArguments.VALID_DATA_LOADER_DROP_LAST,org_col=DataArguments.ORG_TITLE,neg_col=DataArguments.NEGATIVE_TITLE, pin_memory=DataArguments.pin_memory)
    else:
        train_data_loader = create_data_loader(processor = processor, df = train_df, batch_size = ModelArguments.batch_size,
                                               num_workers=ModelArguments.num_workers,
                                               shuffle=DataArguments.TRAIN_DATA_LOADER_SHUFFLE,
                                               drop_last=DataArguments.TRAIN_DATA_LOADER_DROP_LAST,org_col=DataArguments.ORG_TITLE, pin_memory=DataArguments.pin_memory)
        valid_data_loader = create_data_loader(processor = processor, df = valid_df, batch_size = ModelArguments.batch_size,
                                               num_workers=ModelArguments.num_workers,
                                               shuffle=DataArguments.VALID_DATA_LOADER_SHUFFLE,
                                               drop_last=DataArguments.VALID_DATA_LOADER_DROP_LAST,org_col=DataArguments.ORG_TITLE, pin_memory=DataArguments.pin_memory)

    
    if ModelArguments.SCHEDULER_USE:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ModelArguments.LR_SCHEDULER_T_MAX, eta_min=0)

    print(ModelArguments.MODEL_NAME, ModelArguments.TRAIN_MODEL)
    print('Start Training')
    early_stopping = EarlyStopping(patience=ModelArguments.patience, verbose=True)
    save_args(f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_config', early_stopping.best_epoch, early_stopping.best_iteration)
    if ModelArguments.TRAIN_MODEL == 'contrastive':
        print(ModelArguments.Contrastive_Mode)
        
        d_iter = 0
        
        loss_data = []
        for e_idx in range(1, ModelArguments.epochs + 1):
            losses = AverageMeter()

            valid_loss = []

            tbar = tqdm(train_data_loader)
            
            encoder.train()
            for batch_index, batch in enumerate(tbar):
                if early_stopping.early_stop:
                    print("Early stopping Batch")
                    break
                
                d_iter += ModelArguments.batch_size #1
                
                batch = contrastive_tuplify_with_device(batch, device)

                if ModelArguments.Contrastive_Mode == 'imgtotxt':
                    org_input_ids, org_attention_mask, \
                    org_pixel_values = batch

                    outputs = encoder(input_ids=org_input_ids,
                                      attention_mask=org_attention_mask,
                                      pixel_values=org_pixel_values)

                elif ModelArguments.Contrastive_Mode == 'imgtotxt_H':
                    org_input_ids, org_attention_mask,\
                    neg_input_ids, neg_attention_mask, org_pixel_values = batch

                    outputs = encoder(input_ids=torch.cat([org_input_ids, neg_input_ids]),
                                      attention_mask=torch.cat([org_attention_mask, neg_attention_mask]),
                                      pixel_values=org_pixel_values)
                else:
                    print("Eval loss select Error!")
                    exit(-1)
                    
                loss = loss_func(outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), ModelArguments.batch_size)
                tbar.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)       
                
                if d_iter % 20480 == 0 or d_iter == ((len(train_data_loader.dataset)//ModelArguments.batch_size)*ModelArguments.batch_size*ModelArguments.epochs):
                    print('d_iter: ',d_iter)
                    valid_loss = evaluate_CL(encoder, loss_func, valid_data_loader, valid_loss, device) #valid_loss
                    
                    valid_loss_avg = sum(valid_loss) / len(valid_loss)
                    
                    loss_data.append([e_idx, losses.avg, 'Train', d_iter])
                    loss_data.append([e_idx, valid_loss_avg, 'Valid', d_iter])
                
                    print('Valid | epoch: ', str(e_idx), ', Valid Loss Avg: ', str(valid_loss_avg), 'd_iter:', str(d_iter))
                    
                    early_stopping(valid_loss_avg, encoder, e_idx, d_iter)
                    encoder.train() # eval 해제
                    
                if ModelArguments.SCHEDULER_USE:
                    scheduler.step()
                    
            if early_stopping.early_stop:
                print("Early stopping Epoch")
                break
                
        encoder.load_state_dict(torch.load('checkpoint.pt'))
        torch.save(encoder.state_dict(), f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_best.pt')
        save_args(f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_config', early_stopping.best_epoch, early_stopping.best_iteration)
        save_loss(f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_loss', loss_data)
        
    else:
        loss_data = []
        d_iter = 0
        
        for e_idx in range(1, ModelArguments.epochs + 1):
            losses = AverageMeter()

            valid_loss = []

            tbar = tqdm(train_data_loader)

            classifier.train()
            for batch_index, batch in enumerate(tbar):
                if early_stopping.early_stop:
                    print("Early stopping Batch")
                    break
                
                d_iter += ModelArguments.batch_size
                
                batch = classifier_tuplify_with_device(batch, device)

                org_input_ids, org_attention_mask, \
                org_pixel_values, org_label = batch

                outputs = classifier(input_ids=org_input_ids,
                                     attention_mask=org_attention_mask,
                                     pixel_values=org_pixel_values)

                loss = loss_func(outputs.view(-1, 1), org_label.view(-1, 1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), ModelArguments.batch_size)
                tbar.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)

                if d_iter % 20480 == 0 or d_iter == ((len(train_data_loader.dataset)//ModelArguments.batch_size)*ModelArguments.batch_size*ModelArguments.epochs):
                    print('d_iter: ',d_iter)

                    valid_loss, dev_y_preds, dev_y_targets  = evaluate_CF(classifier, loss_func, valid_data_loader, valid_loss, device)
                    
                    valid_loss_avg = sum(valid_loss) / len(valid_loss)

                    loss_data.append([e_idx, losses.avg, 'Train', d_iter])
                    loss_data.append([e_idx, valid_loss_avg, 'Valid', d_iter])

                    print('Valid | epoch: ', str(e_idx), ', Valid Loss Avg: ', str(valid_loss_avg))

                    dev_acc, dev_f1, dev_auroc = compute_accuracy(dev_y_targets, dev_y_preds)

                    print("ACC: {:.4f}, F1_Score: {:.4f}, AUROC: {:.4f}".format(dev_acc, dev_f1, dev_auroc))
                    
                    early_stopping(valid_loss_avg, classifier, e_idx, d_iter)
                    classifier.train() # eval 해제
                    
                if ModelArguments.SCHEDULER_USE == True:
                    scheduler.step()

            if early_stopping.early_stop:
                print("Early stopping Epoch")
                break
                
        classifier.load_state_dict(torch.load('checkpoint.pt'))
        torch.save(classifier.state_dict(), f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_best.pt')
        save_args(f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_config', early_stopping.best_epoch, early_stopping.best_iteration)
        save_loss(f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_loss', loss_data)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    start = time.time()
    main()
    end = time.time()
    print('학습 시간:', end-start)
