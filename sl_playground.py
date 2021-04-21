import argparse

import torch as th
from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right


class DataGenerator(th.utils.data.IterableDataset):
    def __init__(
            self,
            M,
            lambda_real_interval,
            lambda_imag_interval,
    ):
        super().__init__()
        self.M = M
        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.Q = self.coll.Qmat[1:, 1:]
        self.lambda_real_interval = lambda_real_interval
        self.lambda_imag_interval = lambda_imag_interval

    def __iter__(self):
        lam_real_low = self.lambda_real_interval[0]
        lam_real_high = self.lambda_real_interval[1]
        lam_real_interval_size = lam_real_high - lam_real_low

        lam_imag_low = self.lambda_imag_interval[0]
        lam_imag_high = self.lambda_imag_interval[1]
        lam_imag_interval_size = lam_imag_high - lam_imag_low

        while True:
            lam = (
                1 * th.rand((1,)) * lam_real_interval_size + lam_real_low
                + 1j * th.rand((1,)) * lam_imag_interval_size + lam_imag_low
            )
            raise NotImplementedError(
                '`yield` (not `return`!) a tuple like '
                '`(lam, diag_vector_with_min_spectral_radius)` here'
            )
            # yield lam, th.rand(self.M)


class PreconditionerPredictor(th.nn.Module):
    def __init__(self, M):
        super().__init__()
        self.layers = th.nn.Sequential(
            th.nn.Linear(1, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, M),
        )

    def forward(self, inputs):
        return self.layers(inputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--steps',
        type=float,
        default=10000,
        help='Number of learning steps to take.',
    )

    parser.add_argument(
        '--M',
        type=int,
        default=3,
        help=(
            '"Difficulty" of the problem '
            '(proportionally relates to nth-order differential equation)'
            '(choose M = 3 or M = 5 for comparing MIN-preconditioner).'
        ),
    )
    parser.add_argument(
        '--lambda_real_interval',
        type=int,
        nargs=2,
        default=[-100, 0],
        help='Interval to sample the real part of lambda from.',
    )
    parser.add_argument(
        '--lambda_imag_interval',
        type=int,
        nargs=2,
        default=[0, 0],
        help='Interval to sample the imaginary part of lambda from.',
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3E-4,
        help='Learning rate/step size of the model.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Number of samples for each training step',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Base random number seed.',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    th.manual_seed(args.seed)
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

    dataloader = th.utils.data.DataLoader(
        DataGenerator(
            args.M,
            args.lambda_real_interval,
            args.lambda_imag_interval,
        ),
        batch_size=args.batch_size,
    )

    model = PreconditionerPredictor(args.M).to(device)
    loss_func = th.nn.MSELoss()
    opt = th.optim.Adam(model.parameters(), lr=args.learning_rate)
    steps_num_digits = len(str(args.steps))

    for (step, (lams, min_diags)) in enumerate(dataloader):
        lams = lams.float().to(device)
        min_diags = min_diags.to(device)

        diags = model(lams)
        loss = loss_func(diags, min_diags)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f'[{step:>{steps_num_digits}d}/{args.steps}] '
                  f'loss: {loss.item():.5f}')

        if step >= args.steps:
            break


if __name__ == '__main__':
    main()
