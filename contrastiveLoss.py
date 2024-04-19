import torch
import torch.nn.functional as F

class Contrastive_Loss(): 
    def __init__(self, temperature, batch_size):
        self.temperature = temperature
        self.batch_size = batch_size

    def __call__(self, out, do_normalize=True):
        if do_normalize:
            tout = F.normalize(out.text_embeds, dim=1)
            vout = F.normalize(out.image_embeds, dim=1)

        batch_size = int(tout.shape[0] / 2)

        if batch_size != self.batch_size:
            bs = batch_size
        else:
            bs = self.batch_size

        tout_1, tout_2 = tout.split(bs, dim=0) # pos, neg

        sim_matrix_pos = torch.exp(torch.mm(vout, tout_1.t().contiguous()) / self.temperature)
        sim_matrix_neg = torch.exp(torch.mm(vout, tout_2.t().contiguous()) / self.temperature)


        pos_sim = torch.exp(torch.sum(vout * tout_1, dim=-1) / self.temperature)

        loss = (-torch.log(pos_sim / (sim_matrix_pos.sum(dim=-1) + sim_matrix_neg.sum(dim=-1)))).mean()

        return loss