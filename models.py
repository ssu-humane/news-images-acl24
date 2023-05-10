import torch.nn as nn
from transformers import CLIPModel


class ClassifierBaseLineModel(nn.Module):
    def __init__(self, config):
        super(ClassifierBaseLineModel, self).__init__()
        self.config = config
        self.clip = CLIPModel.from_pretrained(config.CLIP_MODEL)
        self.bilinear = nn.Bilinear(config.hidden_size, config.hidden_size, config.intermediate_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(config.intermediate_size, config.n_cls)

    def all_freeze(self):
        for name, child in self.clip.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, pixel_values):
        clip_output = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        x = self.bilinear(clip_output.text_embeds, clip_output.image_embeds)
        x = self.relu(x)
        return self.linear(x)


class ContrastiveBaseLineModel(nn.Module):
    def __init__(self, config):
        super(ContrastiveBaseLineModel, self).__init__()
        self.config = config
        self.clip = CLIPModel.from_pretrained(config.CLIP_MODEL)
        self.text_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def all_freeze(self):
        for name, child in self.clip.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def vision_freeze(self):
        if self.config.CLIP_MODEL == "openai/clip-vit-base-patch32":
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if i == 0 or 197 <= i <= 396:
                    param.requires_grad = False
        else:
            print("vision layer freeze: openai/clip-vit-large-patch14")
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if i == 0 or 197 <= i <= 588:
                    param.requires_grad = False

    def vision_all_text_part_freeze(self):
        if self.config.TEXT_ENCODER_PART_FREEZE == 3:
            print(f"TEXT_ENCODER_PART_FREEZE: {self.config.TEXT_ENCODER_PART_FREEZE}")
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if (0 <= i <= 146) or (197 <= i <= 588): # 3
                    param.requires_grad = False
        elif self.config.TEXT_ENCODER_PART_FREEZE == 6:
            print(f"TEXT_ENCODER_PART_FREEZE: {self.config.TEXT_ENCODER_PART_FREEZE}")
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if (0 <= i <= 98) or (197 <= i <= 588): # 6
                    param.requires_grad = False        
        elif self.config.TEXT_ENCODER_PART_FREEZE == 9:
            print(f"TEXT_ENCODER_PART_FREEZE: {self.config.TEXT_ENCODER_PART_FREEZE}")
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if  (0 <= i <= 50) or (197 <= i <= 588): # 9
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, pixel_values):
        clip_output = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        text_mlp_output = self.text_mlp(clip_output.text_embeds)
        return text_mlp_output, clip_output.image_embeds
    
class CLIPFineT(nn.Module):
    def __init__(self, config):
        super(CLIPFineT, self).__init__()
        self.config = config
        self.clip = CLIPModel.from_pretrained(config.CLIP_MODEL)

    def all_freeze(self):
        for name, child in self.clip.named_children():
            for param in child.parameters():
                param.requires_grad = False
    
    def vision_freeze(self):
        if self.config.CLIP_MODEL == "openai/clip-vit-base-patch32":
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if i == 0 or 197 <= i <= 396:
                    param.requires_grad = False
        else:
            print("vision layer freeze: openai/clip-vit-large-patch14")
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if i == 0 or 197 <= i <= 588:
                    param.requires_grad = False
    def vision_all_text_part_freeze(self):
        if self.config.TEXT_ENCODER_PART_FREEZE == 3:
            print(f"TEXT_ENCODER_PART_FREEZE: {self.config.TEXT_ENCODER_PART_FREEZE}")
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if (0 <= i <= 146) or (197 <= i <= 588): # 3
                    param.requires_grad = False
        elif self.config.TEXT_ENCODER_PART_FREEZE == 6:
            print(f"TEXT_ENCODER_PART_FREEZE: {self.config.TEXT_ENCODER_PART_FREEZE}")
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if (0 <= i <= 98) or (197 <= i <= 588): # 6
                    param.requires_grad = False        
        elif self.config.TEXT_ENCODER_PART_FREEZE == 9:
            print(f"TEXT_ENCODER_PART_FREEZE: {self.config.TEXT_ENCODER_PART_FREEZE}")
            for i, (name, param) in enumerate(self.clip.named_parameters()):
                if  (0 <= i <= 50) or (197 <= i <= 588): # 9
                    param.requires_grad = False
                    
    def forward(self, input_ids, attention_mask, pixel_values):
        clip_output = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        clip_image_embeds = clip_output.image_embeds.detach()
        text_mlp_output = clip_output.text_embeds
        return text_mlp_output, clip_image_embeds#clip_output.image_embeds