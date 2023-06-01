import torch
import numpy as np
from math import pi

L=3
def bbox2gaussian(target, getSize=False):
    center = torch.mean(target, dim=1)
    edge_1 = target[:, 1, :] - target[:, 0, :]
    edge_2 = target[:, 2, :] - target[:, 1, :]
    w = (edge_1 * edge_1).sum(dim=-1, keepdim=True) # (T,1)
    w_ = w.sqrt()
    h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
    diag = torch.cat([w, h], dim=-1).diag_embed() / (4*L*L)
    # cos_=edge_1[:,0].reshape(-1,1)/w_
    # sin_=edge_1[:,1].reshape(-1,1)/w_
    # R=torch.cat([cos_,cos_],dim=-1).diag_embed()+torch.cat([-sin_,sin_],dim=-1).diag_embed()[...,[1,0]]
    cos_sin = edge_1 / w_
    neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
    R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)

    if getSize:
        return (center, R.matmul(diag).matmul(R.transpose(-1, -2))), (w, h)  # (T,d) (T,d,d) (T,1) (T,1)
    else:
        return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))  # (T,d) (T,d,d)


def gaussian2bbox(gmm):
    var = gmm.var
    mu = gmm.mu
    assert mu.size()[1:] == (1, 2)       #(T,1,2)
    assert var.size()[1:] == (1, 2, 2)   #(T,1,2,2)
    T = mu.size()[0]
    var = var.squeeze(1)
    U, s, Vt = torch.svd(var)
    # print(s.size())
    size_half = L * s.sqrt().unsqueeze(1).repeat(1, 4, 1)  #(T,4,2)
    mu = mu.repeat(1, 4, 1)    #(T,4,2)
    dx_dy = size_half*torch.tensor([[-1, 1],
                                  [1, 1],
                                  [1, -1],
                                  [-1, -1]],
                                 dtype=torch.float32, device=size_half.device) # (T,4,2)
    bboxes = (mu+dx_dy.matmul(Vt)).reshape(T, 8)
    return bboxes

class GMM():
    def __init__(self, n_components, n_features=2, mu_init=None, var_init=None, eps=1.e-6, requires_grad=False):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (T, n, d).
        The class owns:
            x:              torch.Tensor (T, n, 1, d)
            mu:             torch.Tensor (T, k, d)
            var:            torch.Tensor (T, k, d) or (T, k, d, d)
            pi:             torch.Tensor (T, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            log_likelihood: torch.Tensor (T, 1)
        args:
            n_components:   int
            n_features:     int
        options:
            mu_init:        torch.Tensor (T, k, d)
            var_init:       torch.Tensor (T, k, d) or (T, k, d, d)
            eps:            float
        """

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.lower_bound_logdet = -783. # lower bound of log_det

        self.requires_grad=requires_grad
        self.T=1
        self.N=9

    def _init_params(self,mu_init=None,var_init=None):
        self.log_likelihood = -np.inf

        if mu_init is not None:
            self.mu_init=mu_init
        if var_init is not None:
            self.var_init=var_init

        if self.requires_grad:
            if self.mu_init is not None:
                assert torch.is_tensor(self.mu_init)
                assert self.mu_init.size() == (self.T, self.n_components,
                                               self.n_features), "Input mu_init does not have required tensor dimensions (%i, %i, %i)" % (
                self.T, self.n_components, self.n_features)
                # (T, k, d)
                self.mu = self.mu_init.clone().requires_grad_().cuda()
            else:
                self.mu = torch.randn((self.T, self.n_components, self.n_features), requires_grad=True).cuda()

            if self.var_init is not None:
                assert torch.is_tensor(self.var_init)
                assert self.var_init.size() == (self.T, self.n_components, self.n_features,
                                                self.n_features), "Input var_init does not have required tensor dimensions (%i, %i, %i, %i)" % \
                                                                  (self.T, self.n_components, self.n_features,
                                                                   self.n_features)
                # (T, k, d, d)
                self.var = self.var_init.clone().requires_grad_().cuda()
            else:
                self.var = torch.eye(self.n_features).reshape((1, 1, self.n_features, self.n_features)).repeat(self.T,
                                                                                                               self.n_components,
                                                                                                               1, 1) \
                    .requires_grad_().cuda()
                # self.var = torch.randn((self.T, self.n_components, self.n_features,
                #                     self.n_features),requires_grad=True).cuda()

            # (T, k, 1)
            self.pi = torch.empty((self.T, self.n_components, 1)).fill_(1. / self.n_components).requires_grad_().cuda()
        else:
            if self.mu_init is not None:
                assert torch.is_tensor(self.mu_init)
                assert self.mu_init.size() == (self.T, self.n_components,
                                               self.n_features), "Input mu_init does not have required tensor dimensions (%i, %i, %i)" % (
                self.T, self.n_components, self.n_features)
                # (T, k, d)
                self.mu = self.mu_init.clone().cuda()
            else:
                self.mu = torch.randn((self.T, self.n_components, self.n_features)).cuda()

            if self.var_init is not None:
                assert torch.is_tensor(self.var_init)
                assert self.var_init.size() == (self.T, self.n_components, self.n_features,
                                                self.n_features), "Input var_init does not have required tensor dimensions (%i, %i, %i, %i)" % \
                                                                  (self.T, self.n_components, self.n_features,
                                                                   self.n_features)
                # (T, k, d, d)
                self.var = self.var_init.clone().cuda()
            else:
                self.var = torch.eye(self.n_features).reshape((1, 1, self.n_features, self.n_features)).repeat(self.T,
                                                                                                               self.n_components,
                                                                                                               1, 1).cuda()
                # self.var = torch.randn((self.T, self.n_components, self.n_features,
                #                     self.n_features),requires_grad=True).cuda()

            # (T, k, 1)
            self.pi = torch.empty((self.T, self.n_components, 1)).fill_(1. / self.n_components).cuda()

        self.params_fitted = False


    def check_size(self, x):
        if len(x.size()) == 3:
            # (T, n, d) --> (T, n, 1, d)
            x = x.unsqueeze(2)

        return x




    def fit(self, x, delta=1e-3, n_iter=10):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (T, n, d) or (T, n, 1, d)
        options:
            delta:      float
            n_iter:     int
        """
        self.T=x.size()[0]
        self.N=x.size()[1]

        select=torch.randint(self.N,size=(self.T*self.n_components,))
        mu_init=x.reshape(-1,self.n_features)[select,:].view(self.T,self.n_components,self.n_features)
        self._init_params(mu_init=mu_init)

        x = self.check_size(x)
        i = 0
        j = np.inf

        while (i <= n_iter) and (not torch.is_tensor(j) or (j >= delta).any()):
            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if (self.log_likelihood.abs() == float("Inf")).any() or (torch.isnan(self.log_likelihood)).any():
                # When the log-likelihood assumes inane values, reinitialize model
                select = torch.randint(self.N, size=(self.T * self.n_components,))
                mu_init = x.reshape(-1, self.n_features)[select, :].view(self.T, self.n_components, self.n_features)
                self._init_params(mu_init=mu_init)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if torch.is_tensor(j) and (j <= delta).any():
                # When score decreases, revert to old parameters
                t = (j <= delta)
                mu_old=t.float().view(self.T,1,1)*mu_old+(~t).float().view(self.T,1,1)*self.mu
                var_old=t.float().view(self.T,1,1,1)*var_old+(~t).float().view(self.T,1,1,1)*self.var
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True




    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (T, n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (T, n, d) or (T, n, 1, d)
        returns:
            log_prob:     torch.Tensor (T, n, k, 1)
        """
        x = self.check_size(x)

        mu = self.mu
        var = self.var
        inverse_var = torch.inverse(var)
        d = x.shape[-1]

        log_2pi = d * np.log(2. * pi)
        det_var = torch.det(var)

        log_det = torch.log(det_var).view(self.T,1,self.n_components,1) # (T,1,k,1)

        # 实验性操作,防止下溢
        log_det[log_det == -np.inf] = self.lower_bound_logdet
        mu = mu.unsqueeze(1)
        x_mu_T = (x - mu).unsqueeze(-2)
        x_mu = (x - mu).unsqueeze(-1)

        # this way reduce memory overhead, but little slow
        # x_mu_T_inverse_var = self._cal_mutmal_x_cov(x_mu_T, inverse_var)
        # x_mu_T_inverse_var_x_mu = self._cal_mutmal_x_x(x_mu_T_inverse_var, x_mu)

        # this way is high memory overhead
        # print(x_mu_T.size(),inverse_var.size())
        x_mu_T_inverse_var = x_mu_T.matmul(inverse_var.unsqueeze(1))
        x_mu_T_inverse_var_x_mu = x_mu_T_inverse_var.matmul(x_mu).squeeze(-1)

        log_p = -.5 * (log_2pi + log_det + x_mu_T_inverse_var_x_mu)

        return log_p


    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (T, n, d) or (T, n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (T, 1)
            log_resp:       torch.Tensor (T, n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi).unsqueeze(1)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=2, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm,dim=(1,2)), log_resp


    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (T, n, d) or (T, n, 1, d)
            log_resp:   torch.Tensor (T, n, k, 1)
        returns:
            pi:         torch.Tensor (T, k, 1)
            mu:         torch.Tensor (T, k, d)
            var:        torch.Tensor (T, k, d, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=1) + self.eps
        mu = torch.sum(resp * x, dim=1) / pi

        # eps = (torch.ones((self.n_features,self.n_features)) * self.eps).to(x.device)
        eps = (torch.eye(self.n_features) * self.eps).to(x.device)
        var = torch.sum((x - mu.unsqueeze(1)).unsqueeze(-1).matmul((x - mu.unsqueeze(1)).unsqueeze(-2)) * resp.unsqueeze(-1), dim=1)\
              / torch.sum(resp, dim=1).unsqueeze(-1) + eps

        pi = pi / x.shape[1]

        return pi, mu, var

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)


        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)


    def __score(self, x, sum_data=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (T, n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (T, 1)
            (or)
            per_sample_score:   torch.Tensor (T, n)

        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi).unsqueeze(1)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=2)

        if sum_data:
            return per_sample_score.sum(dim=1)
        else:
            return per_sample_score.squeeze(-1)


    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() == (self.T, self.n_components,
                                       self.n_features), "Input mu does not have required tensor dimensions (%i, %i, %i)" % (
        self.T, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (self.T, self.n_components, self.n_features):
            self.mu = mu.clone()


    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """

        assert var.size() == (self.T, self.n_components, self.n_features,
                                        self.n_features), "Input var does not have required tensor dimensions (%i, %i, %i, %i)" % \
                                                          (self.T, self.n_components, self.n_features,
                                                           self.n_features)

        if var.size() == (self.n_components, self.n_features, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (self.T, self.n_components, self.n_features, self.n_features):
            self.var = var.clone()

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() == (self.T, self.n_components, 1), "Input pi does not have required tensor dimensions (%i, %i, %i)" % (self.T, self.n_components, 1)

        self.pi = pi.clone()
