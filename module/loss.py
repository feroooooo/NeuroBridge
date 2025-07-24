import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, init_temperature, alpha, beta, eeg_l2norm:bool, img_l2norm:bool, text_l2norm:bool, learnable:bool, softplut:bool):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eeg_l2norm = eeg_l2norm
        self.img_l2norm = img_l2norm
        self.text_l2norm = text_l2norm
        
        self.softplus = softplut
        
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature), requires_grad=learnable)
        self.softplus = nn.Softplus()

    def forward(self, eeg_feature, image_feature, text_feature):
        # L2 normalize embeddings
        if self.eeg_l2norm:
            eeg_feature = F.normalize(eeg_feature, p=2, dim=1)
        if self.img_l2norm:
            image_feature = F.normalize(image_feature, p=2, dim=1)
        if self.beta != 1.0:
            if self.text_l2norm:
                text_feature = F.normalize(text_feature, p=2, dim=1)

        # Calculate similarity matrix (N x N)
        if self.softplus:
            logit_scale = self.softplus(self.logit_scale)
        else:
            logit_scale = torch.exp(self.logit_scale)
        similarity_matrix_ie = torch.matmul(eeg_feature, image_feature.T) * logit_scale
        if self.beta != 1.0:
            similarity_matrix_te = torch.matmul(eeg_feature, text_feature.T) * logit_scale

        # Construct labels
        labels = torch.arange(eeg_feature.shape[0], device=eeg_feature.device)

        # Calculate two parts of the loss
        loss_eeg_ie = self.criterion_cls(similarity_matrix_ie, labels)
        loss_img_ie = self.criterion_cls(similarity_matrix_ie.T, labels)
        if self.beta != 1.0:
            loss_eeg_te = self.criterion_cls(similarity_matrix_te, labels)
            loss_img_te = self.criterion_cls(similarity_matrix_te.T, labels)
            
        if self.alpha != 1.0:
            loss_mse = self.criterion_mse(eeg_feature, image_feature)
        
        # Total loss is the average
        if self.beta != 1.0:
            loss_contrastive_ie = (loss_eeg_ie + loss_img_ie) / 2
            loss_contrastive_te = (loss_eeg_te + loss_img_te) / 2
            loss_contrastive = self.beta * loss_contrastive_ie + (1 - self.beta) * loss_contrastive_te
        else:
            loss_contrastive = (loss_eeg_ie + loss_img_ie) / 2
        
        if self.alpha != 1.0:
            loss = self.alpha * loss_contrastive + (1 - self.alpha) * loss_mse
        else:
            loss = loss_contrastive
        
        return loss