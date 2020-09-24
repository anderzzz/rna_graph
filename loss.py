'''Loss functions for RNA project

'''
import torch

class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class MCRMSELoss(torch.nn.Module):
    def __init__(self, k_scored=(0,1,3)):
        super().__init__()
        self.rmse = RMSELoss()
        self.k_scored = k_scored
        self.num_scored = len(k_scored)

    def forward(self, yhat, y):
        score = 0
        for i in self.k_scored:
            score += self.rmse(yhat[:, i], y[:, i]) / self.num_scored

        return score