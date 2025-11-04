import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Optional, Sequence, List

def gaussian_kernel(x, y, bandwidths=[0.1, 1.0, 10.0]):
    """
    x: (batch_size_1, feature_dim)
    y: (batch_size_2, feature_dim)
    """
    # pairwise_dist_sq = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
    
    x_col = x.unsqueeze(1) # (batch_size_1, 1, feature_dim)
    y_row = y.unsqueeze(0) # (1, batch_size_2, feature_dim)
    pairwise_dist_sq = (x_col - y_row).pow(2).sum(2) # (batch_size_1, batch_size_2)
    
    total_kernel_val = torch.zeros_like(pairwise_dist_sq)
    
    for sigma in bandwidths:
        gamma = 1.0 / (2 * sigma**2)
        total_kernel_val += torch.exp(-gamma * pairwise_dist_sq)

    return total_kernel_val / len(bandwidths)

def mk_mmd_loss(source, target, bandwidths=[0.1, 1.0, 10.0]):
    """
    Multi-Kernel MMD Loss 계산
    source: z_sur (가짜 중간표현), (batch_size, feature_dim)
    target: z (진짜 중간표현), (batch_size, feature_dim)
    """
    
    if source.shape[0] <= 1 or target.shape[0] <= 1:
        return torch.tensor(0.0, device=source.device, requires_grad=True)

    k_ss = gaussian_kernel(source, source, bandwidths)
    k_tt = gaussian_kernel(target, target, bandwidths)
    k_st = gaussian_kernel(source, target, bandwidths)
    
    batch_size = source.shape[0]
    
    # MMD^2 = E[k(s,s)] + E[k(t,t)] - 2*E[k(s,t)]
    
    # k(s,s)의 평균 (대각선 제외)
    # k_ss.sum()은 (B*B)개의 합, k_ss.diag().sum()은 (B)개의 합
    loss_ss = (k_ss.sum() - k_ss.diag().sum()) / (batch_size * (batch_size - 1))
    # k(t,t)의 평균 (대각선 제외)
    loss_tt = (k_tt.sum() - k_tt.diag().sum()) / (batch_size * (batch_size - 1))
    # k(s,t)의 평균 (대각선 개념이 없으므로 그냥 평균)
    loss_st = k_st.mean()

    mmd_loss = loss_ss + loss_tt - 2 * loss_st
    
    return mmd_loss



def mmd_channelwise_loss(source, target, bandwidths=(0.1, 1.0, 10.0)):
    """
    (수정된 버전)
    source, target: (B, C, L)
    반환: 스칼라 (채널별 MMD의 평균)
    """
    B, C, L = source.shape
    xs = source.permute(1, 0, 2).contiguous()
    ys = target.permute(1, 0, 2).contiguous()

    d_xx = torch.cdist(xs, xs, p=2)**2
    d_yy = torch.cdist(ys, ys, p=2)**2
    d_xy = torch.cdist(xs, ys, p=2)**2

    sigmas = torch.tensor(bandwidths, device=xs.device, dtype=xs.dtype) # (M,)
    
    # gamma = 1.0 / (2 * sigma^2)
    # gammas shape: (M,)
    gammas = 1.0 / (2 * sigmas.pow(2)) 
    
    # RBF 커널 계산: (C, B, B, 1) * (M,) -> (C, B, B, M)
    # (sum 대신 mean을 사용하여 여러 커널을 평균냄)
    K_xx = torch.exp(-d_xx.unsqueeze(-1) * gammas).mean(dim=-1) # (C, B, B)
    K_yy = torch.exp(-d_yy.unsqueeze(-1) * gammas).mean(dim=-1) # (C, B, B)
    K_xy = torch.exp(-d_xy.unsqueeze(-1) * gammas).mean(dim=-1) # (C, B, B)

    # unbiased MMD^2 per-channel: (C,)
    # 대각 제거하고 평균
    def offdiag_mean(K):
        B = K.shape[-1]
        # (B, B) bool mask
        mask = ~torch.eye(B, dtype=torch.bool, device=K.device) 
        # K(C,B,B)와 mask(B,B) 브로드캐스팅
        return K.masked_select(mask).view(C, B, B-1).mean(dim=(1,2))

    mmd2_c = offdiag_mean(K_xx) + offdiag_mean(K_yy) - 2.0 * K_xy.mean(dim=(1,2))
    return mmd2_c.mean()
    
class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """
    @acknowledgment:This code is based on the a publicly available code repository.<https://github.com/thuml/Transfer-Learning-Library>
    @author: Junguang Jiang
    """
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks (ICML 2015) <https://arxiv.org/pdf/1502.02791>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as

    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},

    :math:`k` is a kernel function in the function space

    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}

    where :math:`k_{u}` is a single kernel.

    Using kernel trick, MK-MMD can be computed as

    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s})\\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t})\\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}).\\

    Args:
        kernels (tuple(torch.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`

    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels.

    Examples::

        >>> from tllib.modules.kernels import GaussianKernel
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        # print(self.index_matrix.shape)
        # exit()


        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        # print(self.kernels[0](features).shape)
        # exit()

        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix

class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))
    
def mmd_global_loss(z_sur_fake: torch.Tensor, train_representations: torch.Tensor, bandwidths: List[float], linear: bool = False) -> torch.Tensor:
    """
    FORA Trainer 클래스에서 사용된 mmd_global_loss를 래핑합니다.
    (z_sur_fake는 Source, train_representations는 Target 역할을 합니다.)
    """
    z_sur_fake = z_sur_fake.view(z_sur_fake.size(0), -1)
    train_representations = train_representations.view(train_representations.size(0), -1)
    # 1. Multiple Kernels 정의
    # bandwidths 리스트의 각 값은 GaussianKernel의 alpha로 사용됩니다.
    kernels = tuple(GaussianKernel(alpha=alpha, track_running_stats=True) for alpha in bandwidths)
    
    # 2. MK-MMD Loss Module 인스턴스화
    # 배치 크기(z_s.size(0))와 z_t.size(0)가 동일하다고 가정합니다.
    mkmmd_module = MultipleKernelMaximumMeanDiscrepancy(kernels, linear=linear).to(z_sur_fake.device)
    
    # 3. MMD Loss 계산 및 반환
    return mkmmd_module(z_sur_fake, train_representations)
