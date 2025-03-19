import torch

def tanh_log(x):
    return torch.tanh(x) * torch.log(torch.tanh(x) * x + 1)

class MLP(torch.nn.Module):
    '''Just a salt-of-the-earth MLP'''
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        for l in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
            torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        self.nonlinearity = tanh_log

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        h = self.nonlinearity(self.linear3(h))
        h = self.nonlinearity(self.linear4(h))
        return self.linear5(h) 