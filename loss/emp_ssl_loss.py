"""Code for EMP-SSL Loss, largely taken from https://github.com/tsb0601/EMP-SSL"""

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from loss.ap_loss import lossFunction

def tcr_loss(z: Tensor, eps: float) -> Tensor:
    """Total Coding Rate (TCR) loss.

    Args:
        z:
            Patch embeddings.
        eps:
            Epsilon value for numerical stability.

    Returns:
        TCR loss.
    """
    _, batch_size, dim = z.shape
    diag = torch.eye(dim, device=z.device).unsqueeze(0)
    # matmul over batch dimension
    einsum = torch.einsum("vbd,vbe->vde", z, z)
    logdet = torch.logdet(diag + dim / (batch_size * eps) * einsum)
    return 0.5 * logdet.mean()


def invariance_loss(z: Tensor) -> Tensor:
    """Loss representing the similiarity between the patch embeddings and the average of
    the patch embeddings.

    Args:
        z:
            Patch embeddings.
    Returns:
        Similarity loss.
    """
    # z has shape (num_views, batch_size, dim)
    z_mean = z.mean(0, keepdim=True)
    return -F.cosine_similarity(z, z_mean, dim=-1).mean()


class EMPSSLLoss(Module):
    """Implementation of the loss from 'EMP-SSL: Towards Self-Supervised Learning in
    One Training Epoch' [0].

    - [0] EMP-SSL, 2023, https://arxiv.org/abs/2304.03977

    Attributes:
        tcr_eps:
            Total Coding Rate (TCR) epsilon. NOTE: While in the paper, this term is
            squared, we do not square it here as to follow the implementation in the
            official repository.
        inv_coef:
            Coefficient for the invariance loss (Lambda in the paper).

    Examples:
        >>> # initialize loss function
        >>> loss_fn = EMP_SSLLoss()
        >>> base_transform = VICRegViewTransform() # As discussed in paper
        >>> transform_fn = MultiCropTransform(transforms=base_transform, crop_counts=100)
        >>>
        >>> # generate the transformed samples
        >>> samples = transform_fn(image)
        >>>
        >>> # feed through encoder head
        >>> z = torch.cat([model(s) for s in samples])
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(z)
    """

    def __init__(
        self,
        tcr_eps: float = 0.2,
        #inv_coef: float = 200.0,
    ) -> None:
        super().__init__()
        self.tcr_eps = tcr_eps
        #self.inv_coef = inv_coef
        self.speaker_loss = lossFunction(init_w=10.0, init_b=-5.0)
        self.alpha = 0.5

    def forward(self, z_views,z_views_t):
        # z_views has shape (batch_size,num_views, dim)
        feature = []
        #refer_view = z_views_t[0,0,:].unsqueeze(0).unsqueeze(-1)
        batch_size = z_views.size(0)

        #这个for循环实现DINO-FT进行样本筛选，去掉for循环就是DINO-TCR方法，DINO-TCR + DINO-FT = SSDA-TCR
        for i in range(batch_size):
            refer_view = (z_views_t[i, 0, :].unsqueeze(0).unsqueeze(-1) + z_views_t[i, 1, :].unsqueeze(0).unsqueeze(-1))/2
            stu_emb1 = z_views[i,0,:].unsqueeze(0).unsqueeze(-1)
            stu_emb2 = z_views[i, 1, :].unsqueeze(0).unsqueeze(-1)
            score1 = F.cosine_similarity(refer_view, stu_emb1.transpose(0, 2))
            score2 = F.cosine_similarity(refer_view, stu_emb2.transpose(0, 2))
            if score1>=self.alpha and score2>=self.alpha:
                select = torch.stack([stu_emb1,stu_emb2],dim=1).squeeze(-1).squeeze(0)
                feature.append(select)

        new_z_views = torch.stack(feature,dim=0)
        loss1 = self.speaker_loss(new_z_views)
        # z_views = z_views.permute(1,0,2)

        return 0.1*tcr_loss(new_z_views, eps=self.tcr_eps) + loss1#+ self.inv_coef * invariance_loss(z)
