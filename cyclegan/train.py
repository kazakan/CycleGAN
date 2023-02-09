import argparse
from pathlib import Path

from cyclegan.models.model import CycleGANMethod


def train(args):
    trainer = CycleGANMethod(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        n_layers=args.n_layers,
        lr=args.lr,
        b1=args.b1,
        img_size=args.img_size,
        root1=args.root1,
        root2=args.root2,
        batch_size=args.batch_size,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambdaI=args.lambdaI,
        max_epochs=args.max_epochs,
        verbose=args.verbose,
        save_ckpt_interval=args.save_ckpt_interval,
        save_ckpt_dir=args.save_ckpt_dir,
        path_ckpt=args.load_path_ckpt,
        cuda=args.cuda,
    )

    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("root1", type=Path)
    parser.add_argument("root2", type=Path)
    parser.add_argument("save_ckpt_dir", type=Path)

    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)

    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=3)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--b1", type=float, default=0.5)

    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--lambda1", type=float, default=10.0)
    parser.add_argument("--lambda2", type=float, default=10.0)
    parser.add_argument("--lambdaI", type=float, default=0.5)

    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--save-ckpt-interval", type=int, default=50)
    parser.add_argument("--load-path-ckpt", type=Path, default=None)

    parser.add_argument("--verbose", type=int, default=20)

    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        help="Use cuda or not",
    )

    args = parser.parse_args()
    train(args)
