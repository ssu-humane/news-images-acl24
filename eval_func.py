import torch
import numpy as np
from tqdm.auto import tqdm
from utils import (
    contrastive_tuplify_with_device,
    classifier_tuplify_with_device
)
from config import ModelArguments


def evaluate_CL(encoder, loss_func, valid_data_loader, valid_loss, device):
    tbar2 = tqdm(valid_data_loader)
    encoder.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(tbar2):
            batch = contrastive_tuplify_with_device(batch, device)

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

#             elif ModelArguments.Contrastive_Mode == 'txttotxt_H':
#                 org_input_ids, org_attention_mask, \
#                 neg_input_ids, neg_attention_mask, org_pixel_values = batch

#                 outputs = encoder(input_ids=torch.cat([org_input_ids, org_input_ids, neg_input_ids]),
#                                       attention_mask=torch.cat([org_attention_mask, org_attention_mask, neg_attention_mask]),
#                                       pixel_values=org_pixel_values)

#             elif ModelArguments.Contrastive_Mode == 'imgtoimg_H_txttotxt_H':
#                 org_input_ids, org_attention_mask, \
#                 neg_input_ids, neg_attention_mask, org_pixel_values = batch

#                 outputs = encoder(input_ids=torch.cat([org_input_ids, org_input_ids, neg_input_ids]),
#                                   attention_mask=torch.cat(
#                                       [org_attention_mask, org_attention_mask, neg_attention_mask]),
#                                   pixel_values=org_pixel_values)
            else:
                print("Eval loss select Error!")
                exit(-1)
            loss = loss_func(outputs)
            valid_loss.append(loss.item())
            
    return valid_loss

def evaluate_CF(classifier, loss_func, valid_data_loader, valid_loss, device):
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

            y_pred = torch.sigmoid(outputs) # 확률로 변환
            y_pred = y_pred.detach().cpu().numpy()
            org_label = org_label.to('cpu').numpy()
            dev_y_preds.append(y_pred)
            dev_y_targets.append(org_label)

    dev_y_preds = np.concatenate(dev_y_preds).reshape((-1,))
    dev_y_targets = np.concatenate(dev_y_targets).reshape((-1,)).astype(int)
            
    return valid_loss, dev_y_preds, dev_y_targets
