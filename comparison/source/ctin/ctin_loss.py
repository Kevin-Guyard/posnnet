import torch


class CTINLoss(torch.nn.Module):
    def __init__(self, dt=0.01, init_log_sigma_v=0.0, init_log_sigma_c=0.0):
        super().__init__()
        self.dt = dt
        # log σ_v and log σ_c; σ^2 = exp(2 * logσ)
        self.log_sigma_v = torch.nn.Parameter(torch.tensor(init_log_sigma_v))
        self.log_sigma_c = torch.nn.Parameter(torch.tensor(init_log_sigma_c))

    def forward(self, y_vel_pred, y_cov_pred, y_vel, y_pos):
        # ----- IVL: position term -----
        delta_pos_pred = y_vel_pred * self.dt
        pos_pred = torch.cumsum(delta_pos_pred, dim=1)
        pos_err_sq = (pos_pred - y_pos).pow(2).sum(dim=-1)
        L_v_p = pos_err_sq.mean()

        # ----- IVL: velocity term -----
        vel_err_sq = (y_vel_pred - y_vel).pow(2).sum(dim=-1)
        L_v_e = vel_err_sq.mean()

        L_v = L_v_p + L_v_e

        # ----- CNL -----
        eps = 1e-6
        var = torch.nn.functional.softplus(y_cov_pred) + eps  # (B, T, 3)
        diff = y_vel - y_vel_pred
        maha = (diff.pow(2) / var).sum(dim=-1)
        logdet = torch.log(var).sum(dim=-1)
        L_c = 0.5 * (maha + logdet)
        L_c = L_c.mean()

        # ----- Multi-task combination (Eq. 7) -----
        # δ_v = exp(log_sigma_v), δ_c = exp(log_sigma_c)
        inv_var_v = torch.exp(-2.0 * self.log_sigma_v)  # 1 / δ_v^2
        inv_var_c = torch.exp(-2.0 * self.log_sigma_c)  # 1 / δ_c^2

        loss = 0.5 * (inv_var_v * L_v + inv_var_c * L_c) + (self.log_sigma_v + self.log_sigma_c)

        return loss
