import torch.nn as nn
from transformers import CLIPModel


class ClassifierBaseLineModel(nn.Module):
    def __init__(self, config):
        super(ClassifierBaseLineModel, self).__init__()
        self.config = config
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
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
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
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

    def forward(self, input_ids, attention_mask, pixel_values):
        clip_output = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        text_mlp_output = self.text_mlp(clip_output.text_embeds)
        return text_mlp_output, clip_output.image_embeds