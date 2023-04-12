import torch
import torch.nn.functional as F

############################
"""
loss name 각 case 별로 변경
"""
############################

class Contrastive_Loss1(): # Baseline, image to text
    def __init__(self, temperature, batch_size):
        self.temperature = temperature
        self.batch_size = batch_size

    def __call__(self, out, do_normalize=True):
        if do_normalize:
            tout = F.normalize(out[0], dim=1)
            vout = F.normalize(out[1], dim=1)

        sim_matrix = torch.exp(torch.mm(vout, tout.t().contiguous()) / self.temperature)  # 이미지와 텍스트

        pos_sim = torch.exp(torch.sum(vout * tout, dim=-1) / self.temperature)  # pos 이미지, 텍스트

        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return loss

class Contrastive_Loss2(): # Baseline + hardnegative, image to text
    def __init__(self, temperature, batch_size):
        self.temperature = temperature
        self.batch_size = batch_size

    def __call__(self, out, do_normalize=True):
        if do_normalize:
            tout = F.normalize(out[0], dim=1)
            vout = F.normalize(out[1], dim=1)

        batch_size = int(tout.shape[0] / 2)

        if batch_size != self.batch_size:
            bs = batch_size
        else:
            bs = self.batch_size

        tout_1, tout_2 = tout.split(bs, dim=0) # pos, neg

        sim_matrix_pos = torch.exp(torch.mm(vout, tout_1.t().contiguous()) / self.temperature)  # 이미지와 텍스트
        sim_matrix_neg = torch.exp(torch.mm(vout, tout_2.t().contiguous()) / self.temperature)  # 이미지와 텍스트


        pos_sim = torch.exp(torch.sum(vout * tout_1, dim=-1) / self.temperature)

        loss = (-torch.log(pos_sim / (sim_matrix_pos.sum(dim=-1) + sim_matrix_neg.sum(dim=-1)))).mean()

        return loss

class Contrastive_Loss3(): # text to text, hardnegative
    def __init__(self, temperature, batch_size):
        self.temperature = temperature
        self.batch_size = batch_size

    def __call__(self, out, do_normalize=True):
        if do_normalize:
            out = F.normalize(out[0], dim=1)

        batch_size = int(out.shape[0] / 3)

        if batch_size != self.batch_size:
            bs = batch_size
        else:
            bs = self.batch_size

        out_1, out_2, out_3 = out.split(bs, dim=0)  # pos, pos, neg

        sim_matrix_pos = torch.exp(torch.mm(out_1, out_2.t().contiguous()) / self.temperature)  # 텍스트와 텍스트
        sim_matrix_neg = torch.exp(torch.mm(out_1, out_3.t().contiguous()) / self.temperature)  # 텍스트와 텍스트


        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)  # 텍스트 pos, 텍스트

        loss = (-torch.log(pos_sim / (sim_matrix_pos.sum(dim=-1)+sim_matrix_neg.sum(dim=-1)))).mean()

        return loss

class Contrastive_Loss4(): # image to text + text to text
    def __init__(self, temperature, gamma, batch_size):
        self.temperature = temperature
        self.gamma = gamma
        self.batch_size = batch_size

    def __call__(self, out, do_normalize=True):
        if do_normalize:
            tout = F.normalize(out[0], dim=1)
            vout = F.normalize(out[1], dim=1)

        batch_size = int(tout.shape[0] / 3)

        if batch_size != self.batch_size:
            bs = batch_size
        else:
            bs = self.batch_size

        tout_1, tout_2, tout_3 = tout.split(bs, dim=0)  # pos, pos, neg


        sim_matrix_img_to_txt_pos = torch.exp(torch.mm(vout, tout_2.t().contiguous()) / self.temperature)  # 이미지와 텍스트
        sim_matrix_img_to_txt_neg = torch.exp(torch.mm(vout, tout_3.t().contiguous()) / self.temperature)  # 이미지와 텍스트

        sim_matrix_txt_to_txt_pos = torch.exp(torch.mm(tout_1, tout_2.t().contiguous()) / self.temperature)  # 텍스트와 텍스트
        sim_matrix_txt_to_txt_neg = torch.exp(torch.mm(tout_1, tout_3.t().contiguous()) / self.temperature)  # 텍스트와 텍스트


        pos_sim_img_to_txt = torch.exp(torch.sum(vout * tout_2, dim=-1) / self.temperature)  # pos 이미지, 텍스트
        pos_sim_txt_to_txt = torch.exp(torch.sum(tout_1 * tout_2, dim=-1) / self.temperature)  # pos 텍스트, 텍스트

        img_to_txt_loss = -torch.log(pos_sim_img_to_txt / (sim_matrix_img_to_txt_pos.sum(dim=-1) + sim_matrix_img_to_txt_neg.sum(dim=-1)))
        txt_to_txt_loss = -torch.log(pos_sim_txt_to_txt / (sim_matrix_txt_to_txt_pos.sum(dim=-1) + sim_matrix_txt_to_txt_neg.sum(dim=-1)))

        loss = ((img_to_txt_loss + (self.gamma*txt_to_txt_loss)).mean())/2

        return loss