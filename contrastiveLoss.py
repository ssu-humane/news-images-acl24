import torch
import torch.nn.functional as F

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


class Contrastive_Loss3(): # Baseline + hardnegative, image to text
    def __init__(self, temperature, batch_size):
        self.temperature = temperature
        self.batch_size = batch_size

    def __call__(self, out, do_normalize=True):
        if do_normalize:
            tout = F.normalize(out[0], dim=1)
            vout = F.normalize(out[1], dim=1)

        batch_size = int(tout.shape[0] / 4)

        if batch_size != self.batch_size:
            bs = batch_size
        else:
            bs = self.batch_size

        tout_1, tout_2, tout_3, tout_4 = tout.split(bs, dim=0) # pos, neg(NP+N+V+P), neg(N), neg(V)

        sim_matrix_pos = torch.exp(torch.mm(vout, tout_1.t().contiguous()) / self.temperature)  # 이미지와 텍스트
        sim_matrix_neg1 = torch.exp(torch.mm(vout, tout_2.t().contiguous()) / self.temperature)  # 이미지와 텍스트
        sim_matrix_neg2 = torch.exp(torch.mm(vout, tout_3.t().contiguous()) / self.temperature)  # 이미지와 텍스트
        sim_matrix_neg3 = torch.exp(torch.mm(vout, tout_4.t().contiguous()) / self.temperature)  # 이미지와 텍스트

        pos_sim = torch.exp(torch.sum(vout * tout_1, dim=-1) / self.temperature)

        loss = (-torch.log(pos_sim / (sim_matrix_pos.sum(dim=-1) + sim_matrix_neg1.sum(dim=-1) + sim_matrix_neg2.sum(dim=-1) + sim_matrix_neg3.sum(dim=-1)))).mean()

        return loss
    
class Contrastive_Loss4(): # Baseline + hardnegative, image to text
    def __init__(self, temperature, batch_size):
        self.temperature = temperature
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

        tout_1, tout_2, tout_3 = tout.split(bs, dim=0) # pos, neg(NP+N+V+P), neg(N), neg(V)

        sim_matrix_pos = torch.exp(torch.mm(vout, tout_1.t().contiguous()) / self.temperature)  # 이미지와 텍스트
        sim_matrix_neg1 = torch.exp(torch.mm(vout, tout_2.t().contiguous()) / self.temperature)  # 이미지와 텍스트
        sim_matrix_neg2 = torch.exp(torch.mm(vout, tout_3.t().contiguous()) / self.temperature)  # 이미지와 텍스트

        pos_sim = torch.exp(torch.sum(vout * tout_1, dim=-1) / self.temperature)

        loss = (-torch.log(pos_sim / (sim_matrix_pos.sum(dim=-1) + sim_matrix_neg1.sum(dim=-1) + sim_matrix_neg2.sum(dim=-1)))).mean()

        return loss
    

class Contrastive_text_to_image_Loss(): # Baseline + hardnegative, image to text
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

        sim_matrix_pos = torch.exp(torch.mm(tout_1, vout.t().contiguous()) / self.temperature)  # 이미지와 텍스트
        sim_matrix_neg = torch.exp(torch.mm(tout_2, vout.t().contiguous()) / self.temperature)  # 이미지와 텍스트

        pos_sim = torch.exp(torch.sum(tout_1 * vout, dim=-1) / self.temperature)

        loss = (-torch.log(pos_sim / (sim_matrix_pos.sum(dim=-1) + sim_matrix_neg.sum(dim=-1)))).mean()

        return loss

class Contrastive_i2t_t2i_Loss(): # image to text + text to text
    def __init__(self, temperature, batch_size):
        self.temperature = temperature
        # self.gamma = gamma
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

        tout_1, tout_2 = tout.split(bs, dim=0)  # pos, neg

        sim_matrix_img_to_txt_pos = torch.exp(torch.mm(vout, tout_1.t().contiguous()) / self.temperature)  # 이미지와 텍스트
        sim_matrix_img_to_txt_neg = torch.exp(torch.mm(vout, tout_2.t().contiguous()) / self.temperature)  # 이미지와 텍스트

        sim_matrix_txt_to_img_pos = torch.exp(torch.mm(tout_1, vout.t().contiguous()) / self.temperature)  # 텍스트와 이미지
        sim_matrix_txt_to_img_neg = torch.exp(torch.mm(tout_2, vout.t().contiguous()) / self.temperature)  # 텍스트와 이미지

        pos_sim_img_to_txt = torch.exp(torch.sum(vout * tout_1, dim=-1) / self.temperature)  # pos 이미지, 텍스트
        pos_sim_txt_to_img = torch.exp(torch.sum(tout_1 * vout, dim=-1) / self.temperature)  # pos 텍스트, 이미지

        img_to_txt_loss = (-torch.log(pos_sim_img_to_txt / (sim_matrix_img_to_txt_pos.sum(dim=-1) + sim_matrix_img_to_txt_neg.sum(dim=-1)))).mean()
        txt_to_img_loss = (-torch.log(pos_sim_txt_to_img / (sim_matrix_txt_to_img_pos.sum(dim=-1) + sim_matrix_txt_to_img_neg.sum(dim=-1)))).mean()

        loss = (img_to_txt_loss + txt_to_img_loss)/2

        return loss