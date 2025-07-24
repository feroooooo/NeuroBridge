import torch.nn as nn

class ProjectorLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectorLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


class ProjectorMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectorMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class ProjectorDirect(nn.Module):
    def __init__(self):
        super(ProjectorDirect, self).__init__()

    def forward(self, x):
        return x