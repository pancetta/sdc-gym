import torch as th


class Swish(th.nn.Module):
    def forward(self, x):
        return x * th.nn.functional.sigmoid(x)


class Mish(th.nn.Module):
    def forward(self, x):
        return x * th.tanh(th.nn.functional.softplus(x))
