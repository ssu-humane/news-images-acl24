import torch.nn as nn
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from utils import Output

class ContrastiveBaseLineModel(nn.Module):
    def __init__(self, config):
        super(ContrastiveBaseLineModel, self).__init__()
        self.config = config
        self.text_model =  CLIPTextModelWithProjection.from_pretrained(self.config.CLIP_MODEL)
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.config.CLIP_MODEL)

    def vision_freeze(self):
        if self.config.VISION_ENCODER_FREEZE == 'all':
            print('vision all freeze')
            for idx, (name, param) in enumerate(self.vision_model.named_parameters()):
                param.requires_grad = False
        elif self.config.VISION_ENCODER_FREEZE == 0:
            print('vision 0 layer freeze')
        elif 1 <= self.config.VISION_ENCODER_FREEZE <= 23:
            freeze_num = 16 * self.config.VISION_ENCODER_FREEZE + 4
            print(f'vision {self.config.VISION_ENCODER_FREEZE} layer freeze')
            for idx, (name, param) in enumerate(self.vision_model.named_parameters()):
                if 0 <= idx <= freeze_num:
                    param.requires_grad = False
        elif self.config.VISION_ENCODER_FREEZE == 24:
            print('vision 24 layer freeze')
            for idx, (name, param) in enumerate(self.vision_model.named_parameters()):
                if 0 <= idx <= 390:
                    param.requires_grad = False
    
    def text_freeze(self):
        if self.config.TEXT_ENCODER_FREEZE == 'all':
            print('text all layer freeze')
            for idx, (name, param) in enumerate(self.text_model.named_parameters()):
                param.requires_grad = False
        elif self.config.TEXT_ENCODER_FREEZE == 0:
            print('text 0 layer freeze')
        elif 1 <= self.config.TEXT_ENCODER_FREEZE <= 11:
            freeze_num = 16 * self.config.TEXT_ENCODER_FREEZE + 1 
            print(f'text {self.config.TEXT_ENCODER_FREEZE} layer freeze')
            for idx, (name, param) in enumerate(self.text_model.named_parameters()):
                if 0 <= idx <= freeze_num:
                    param.requires_grad = False
        elif self.config.TEXT_ENCODER_FREEZE == 12:
            print('text 12 layer freeze')
            for idx, (name, param) in enumerate(self.text_model.named_parameters()):
                if 0 <= idx <= 195:
                    param.requires_grad = False
            

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        text_embeds = text_outputs.text_embeds
        image_embeds = vision_outputs.image_embeds

        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        return Output(
            text_embeds=text_embeds,
            image_embeds=image_embeds
        )
        