import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from transformers import CLIPProcessor
from models import ClassifierBaseLineModel, ContrastiveBaseLineModel
from utils import (
    create_data_loader,
    set_seed,
    AverageMeter,
    compute_accuracy,
    classifier_tuplify_with_device,
    contrastive_tuplify_with_device
)
from config import ModelArguments, DataArguments, save_args
from contrastiveLoss import (
    Contrastive_Loss1,
    Contrastive_Loss2,
    Contrastive_Loss3,
    Contrastive_Loss4
)


def main():
    set_seed(ModelArguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if ModelArguments.TRAIN_MODEL == 'contrastive':
        encoder = ContrastiveBaseLineModel(ModelArguments)
        encoder.all_freeze()
        encoder = nn.DataParallel(encoder)
        encoder = encoder.to(device)

        if ModelArguments.Contrastive_Mode == 'imgtotxt':
            loss_func = Contrastive_Loss1(temperature = ModelArguments.LOSS_TEMPERATURE,
                                          batch_size = ModelArguments.batch_size)
        elif ModelArguments.Contrastive_Mode == 'imgtotxt_H':
            loss_func = Contrastive_Loss2(temperature=ModelArguments.LOSS_TEMPERATURE,
                                          batch_size=ModelArguments.batch_size)
        elif ModelArguments.Contrastive_Mode == 'txttotxt_H':
            loss_func = Contrastive_Loss3(temperature=ModelArguments.LOSS_TEMPERATURE,
                                          batch_size=ModelArguments.batch_size)
        elif ModelArguments.Contrastive_Mode == 'imgtoimg_H_txttotxt_H':
            loss_func = Contrastive_Loss4(temperature=ModelArguments.LOSS_TEMPERATURE,
                                          gamma = ModelArguments.GAAMA,
                                          batch_size=ModelArguments.batch_size)
        else:
            print("Contrastive Loss select error")

        optimizer = optim.AdamW(encoder.parameters(), lr=ModelArguments.learning_rate)

        if ModelArguments.SCHEDULER_USE == True:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ModelArguments.LR_SCHEDULER_T_MAX, eta_min=0)

    elif ModelArguments.TRAIN_MODEL == 'classifier':
        classifier = ClassifierBaseLineModel(ModelArguments)
        classifier.all_freeze()
        classifier = nn.DataParallel(classifier)
        classifier = classifier.to(device)

        loss_func = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(classifier.parameters(), lr=ModelArguments.learning_rate)

        if ModelArguments.SCHEDULER_USE == True:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ModelArguments.LR_SCHEDULER_T_MAX, eta_min=0)
    else:
        print("Train Model select error")

    train_df = pd.read_json(DataArguments.TRAIN_DATA_PATH)
    valid_df = pd.read_json(DataArguments.VALID_DATA_PATH)

    train_data_loader = create_data_loader(processor =processor, df = train_df, batch_size = ModelArguments.batch_size,
                                           num_workers=ModelArguments.num_workers,
                                           shuffle=DataArguments.DATA_LOADER_SHUFFLE,
                                           drop_last=DataArguments.DATA_LOADER_DROP_LAST)
    valid_data_loader = create_data_loader(processor = processor, df = valid_df, batch_size = ModelArguments.batch_size,
                                           num_workers=ModelArguments.num_workers,
                                           shuffle=DataArguments.DATA_LOADER_SHUFFLE,
                                           drop_last=DataArguments.DATA_LOADER_DROP_LAST)


    print(ModelArguments.MODEL_NAME, ModelArguments.TRAIN_MODEL)

    print('Start Training')
    if ModelArguments.TRAIN_MODEL == 'contrastive':
        print(ModelArguments.Contrastive_Mode)
        
        d_iter = 0
        stop = False
        
        loss_data = []

        for e_idx in range(1, ModelArguments.epochs + 1):
            if stop == True:
                break
                
            losses = AverageMeter()

            valid_loss = []

            tbar = tqdm(train_data_loader)

            encoder.train()
            for batch_index, batch in enumerate(tbar):
                if d_iter >= ModelArguments.iteration:
                    stop = True
                    break
                
                d_iter += ModelArguments.batch_size
                
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

                elif ModelArguments.Contrastive_Mode == 'txttotxt_H':
                    org_input_ids, org_attention_mask, \
                    neg_input_ids, neg_attention_mask = batch

                    outputs = encoder(input_ids=torch.cat([org_input_ids, org_input_ids, neg_input_ids]),
                                      attention_mask=torch.cat([org_attention_mask, org_attention_mask, neg_attention_mask]))

                else:  # ModelArguments.Contrastive_Mode == 'imgtoimg_H_txttotxt_H':
                    org_input_ids, org_attention_mask, \
                    neg_input_ids, neg_attention_mask, org_pixel_values = batch

                    outputs = encoder(input_ids=torch.cat([org_input_ids, org_input_ids, neg_input_ids]),
                                      attention_mask=torch.cat([org_attention_mask, org_attention_mask, neg_attention_mask]),
                                      pixel_values=org_pixel_values)

                loss = loss_func(outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), ModelArguments.batch_size)
                tbar.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)
                
                del batch, outputs, loss

            loss_data.append([e_idx, losses.avg, 'Train'])
            
            if stop == True:
                break
            
            tbar2 = tqdm(valid_data_loader)
            encoder.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(tbar2):
                    batch = contrastive_tuplify_with_device(batch, device)  # 각 모델 별로 return 다른거 적용하여 수정

                    if ModelArguments.Contrastive_Mode == 'imgtotxt':
                        org_input_ids, org_attention_mask, \
                        org_pixel_values = batch

                        outputs = encoder(input_ids=org_input_ids,
                                          attention_mask=org_attention_mask,
                                          pixel_values=org_pixel_values)

                    elif ModelArguments.Contrastive_Mode == 'imgtotxt_H':
                        org_input_ids, org_attention_mask, \
                        neg_input_ids, neg_attention_mask, org_pixel_values = batch

                        outputs = encoder(input_ids=torch.cat([org_input_ids, neg_input_ids]),
                                          attention_mask=torch.cat([org_attention_mask, neg_attention_mask]),
                                          pixel_values=org_pixel_values)

                    elif ModelArguments.Contrastive_Mode == 'txttotxt_H':
                        org_input_ids, org_attention_mask, \
                        neg_input_ids, neg_attention_mask = batch

                        outputs = encoder(input_ids=torch.cat([org_input_ids, neg_input_ids]),
                                          attention_mask=torch.cat([org_attention_mask, neg_attention_mask]))

                    else:  # ModelArguments.Contrastive_Mode == 'imgtoimg_H_txttotxt_H':
                        org_input_ids, org_attention_mask, \
                        neg_input_ids, neg_attention_mask, org_pixel_values = batch

                        outputs = encoder(input_ids=torch.cat([org_input_ids, org_input_ids, neg_input_ids]),
                                          attention_mask=torch.cat(
                                              [org_attention_mask, org_attention_mask, neg_attention_mask]),
                                          pixel_values=org_pixel_values)

                    loss = loss_func(outputs)
                    valid_loss.append(loss.item())
                    
                    del batch, outputs, loss
                    
                if ModelArguments.SCHEDULER_USE == True:
                    scheduler.step()
                    
                valid_loss_avg = sum(valid_loss) / len(valid_loss)
                loss_data.append([e_idx, valid_loss_avg, 'Valid'])

                print('Valid | epoch: ', str(e_idx), ', Valid Loss Avg: ', str(valid_loss_avg), 'd_iter:', str(d_iter))

            if e_idx % 10 == 0:
                torch.save(encoder.state_dict(), f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_{e_idx}.pt')
        torch.save(encoder.state_dict(), f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}_final.pt')

    else:
        loss_data = []

        best_ep = 0
        best_model_state_on_dev = None
        best_dev_acc = 0.0
        best_dev_f1 = 0.0
        best_dev_auroc = 0.0

        for e_idx in range(1, ModelArguments.epochs + 1):
            losses = AverageMeter()

            valid_loss = []

            tbar = tqdm(train_data_loader)

            classifier.train()
            for batch_index, batch in enumerate(tbar):
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

            loss_data.append([e_idx, losses.avg, 'Train'])

            tbar2 = tqdm(valid_data_loader)
            classifier.eval()
            with torch.no_grad():
                dev_y_preds, dev_y_targets = [], []

                for batch_index, batch in enumerate(tbar2):
                    batch = classifier_tuplify_with_device(batch, device)

                    org_input_ids, org_attention_mask, \
                    org_pixel_values, org_label = batch

                    outputs = classifier(input_ids=org_input_ids,
                                         attention_mask=org_attention_mask,
                                         pixel_values=org_pixel_values)

                    loss = loss_func(outputs.view(-1, 1), org_label.view(-1, 1))
                    valid_loss.append(loss.item())

                    y_pred = torch.sigmoid(outputs)
                    y_pred = y_pred.detach().cpu().numpy()
                    org_label = org_label.to('cpu').numpy()
                    dev_y_preds.append(y_pred)
                    dev_y_targets.append(org_label)

                if ModelArguments.SCHEDULER_USE == True:
                    scheduler.step()
                    
                valid_loss_avg = sum(valid_loss) / len(valid_loss)
                loss_data.append([e_idx, valid_loss_avg, 'Valid'])

                print('Valid | epoch: ', str(e_idx), ', Valid Loss Avg: ', str(valid_loss_avg))

                dev_y_preds = np.concatenate(dev_y_preds).reshape((-1,))
                dev_y_targets = np.concatenate(dev_y_targets).reshape((-1,)).astype(int)
                dev_acc, dev_f1, dev_auroc = compute_accuracy(dev_y_targets, dev_y_preds)

                print("ACC: {:.4f}, F1_Score: {:.4f}, AUROC: {:.4f}".format(dev_acc, dev_f1, dev_auroc))

            if dev_acc > best_dev_acc:
                best_ep = e_idx
                best_dev_acc = dev_acc
                best_dev_f1 = dev_f1
                best_dev_auroc = dev_auroc
                best_model_state_on_dev = classifier.state_dict()
        print("Best Model\nEpoch: {}\nACC: {:.4f}\nF1_Score: {:.4f}\nAUROC: {:.4f}".format(best_ep, best_dev_acc,
                                                                                           best_dev_f1, best_dev_auroc))
        torch.save(best_model_state_on_dev, f'{ModelArguments.MODEL_SAVE_PATH}/{ModelArguments.MODEL_NAME}.pt')
    save_args(ModelArguments.MODEL_SAVE_PATH, loss_data)


if __name__ == "__main__":
    main()