import torch


class MLP(torch.nn.Module):

    def __init__(self, obs_size,n_act):
        super().__init__()
        self.mlp = self.__mlp(obs_size,n_act)

    def __mlp(self,obs_size,n_act):
        return torch.nn.Sequential(
            torch.nn.Linear(obs_size, 50),
            # self.fc1.weight.data.normal_(0, 0.1),  # 权重初始化 (均值为0，方差为0.1的正态分布)
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, n_act)
        )

    def forward(self, x):
        return self.mlp(x)

