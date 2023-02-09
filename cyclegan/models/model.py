import itertools
import os
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from cyclegan.dataset.img_folder import UnpairedImageDataset
from cyclegan.models.loss import (CycleGANDiscriminatorLoss,
                                  CycleGANGeneratorLoss)
from cyclegan.models.units import Discriminator, Generator
from cyclegan.utils.logger import CsvLogger, SimpleLogger
from cyclegan.utils.meter import MultiAverageMeter


class CycleGANMethod:
    def __init__(
        self,
        root1: os.PathLike,
        root2: os.PathLike,
        save_ckpt_dir: os.PathLike,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden: int = 64,
        n_blocks: int = 6,
        n_layers: int = 3,
        lr: float = 0.002,
        b1: float = 0.1,
        img_size: int = 256,
        batch_size: int = 32,
        lambda1: float = 0.001,
        lambda2: float = 0.001,
        lambdaI: float = 0.001,
        max_epochs: int = 500,
        save_ckpt_interval: int = 10,
        path_ckpt: Optional[os.PathLike] = None,
        cuda: bool = False,
        verbose: int = -1,
    ):
        self.save_ckpt_interval = save_ckpt_interval
        self.save_ckpt_dir = Path(save_ckpt_dir)
        self.cuda = cuda
        self.dev = "cuda" if cuda else "cpu"

        if not self.save_ckpt_dir.exists():
            self.save_ckpt_dir.mkdir()

        self.verbose = verbose if verbose > 0 else save_ckpt_interval
        self.logcols = ["epoch", "loss_G", "loss_D_A", "loss_D_B"]
        self.csvlogger = CsvLogger(self.logcols, self.save_ckpt_dir / "history.csv")
        self.stringlogger = SimpleLogger(self.logcols)

        if path_ckpt is None:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.hidden = hidden
            self.n_blocks = n_blocks
            self.n_layers = n_layers
            self.lr = lr
            self.b1 = b1
            self.img_size = img_size
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.lambdaI = lambdaI
            self.max_epochs = max_epochs
            self.cur_epoch = 0
        else:
            state_dict = torch.load(path_ckpt, map_location=self.dev)
            self.in_channels = state_dict["in_channels"]
            self.out_channels = state_dict["out_channels"]
            self.hidden = state_dict["hidden"]
            self.n_blocks = state_dict["n_blocks"]
            self.n_layers = state_dict["n_layers"]
            self.lr = state_dict["lr"]
            self.b1 = state_dict["b1"]
            self.img_size = state_dict["img_size"]
            self.lambda1 = state_dict["lambda1"]
            self.lambda2 = state_dict["lambda2"]
            self.lambdaI = state_dict["lambdaI"]
            self.max_epochs = state_dict["max_epochs"]
            self.cur_epoch = state_dict["cur_epoch"]

        self.G_A = Generator(
            self.in_channels, self.out_channels, self.hidden, self.n_blocks
        )
        self.G_B = Generator(
            self.in_channels, self.out_channels, self.hidden, self.n_blocks
        )

        # discriminator
        self.D_A = Discriminator(
            self.in_channels, self.out_channels, self.hidden, self.n_layers
        )
        self.D_B = Discriminator(
            self.in_channels, self.out_channels, self.hidden, self.n_layers
        )

        # optimizer
        self.optim_G = torch.optim.AdamW(
            itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
            lr=self.lr,
            betas=(self.b1, 0.999),
        )
        self.optim_D = torch.optim.AdamW(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.lr,
            betas=(self.b1, 0.999),
        )
        self.lr_sched_G = torch.optim.lr_scheduler.StepLR(self.optim_G, 100, 0.8)
        self.lr_sched_D = torch.optim.lr_scheduler.StepLR(self.optim_D, 100, 0.8)

        # train data
        transforms = T.Compose(
            [
                T.Resize((self.img_size + 30, self.img_size + 30)),
                T.RandomCrop((self.img_size, self.img_size)),
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.RandomRotation(90),
                T.ToTensor(),
            ]
        )
        self.train_dataset = UnpairedImageDataset(
            Path(root1), Path(root2), transform=transforms
        )
        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=batch_size
        )

        # loss func
        self.cyclegan_generator_loss_func = CycleGANGeneratorLoss(
            self.lambda1, self.lambda2, self.lambdaI
        )
        self.cyclegan_discriminator_loss_func = CycleGANDiscriminatorLoss()

        if path_ckpt is not None:
            self.G_A.load_state_dict(state_dict["G_A_state_dict"])
            self.G_B.load_state_dict(state_dict["G_B_state_dict"])
            self.D_A.load_state_dict(state_dict["D_A_state_dict"])
            self.D_B.load_state_dict(state_dict["D_B_state_dict"])
            self.optim_G.load_state_dict(state_dict["optim_G_state_dict"])
            self.optim_D.load_state_dict(state_dict["optim_D_state_dict"])
            self.lr_sched_G.load_state_dict(state_dict["lr_sched_G_state_dict"])
            self.lr_sched_D.load_state_dict(state_dict["lr_sched_D_state_dict"])

    def train(self):
        self.G_A.to(self.dev)
        self.G_B.to(self.dev)
        self.D_A.to(self.dev)
        self.D_B.to(self.dev)

        self.G_A.train()
        self.G_B.train()

        # check for ckpt directory
        if self.save_ckpt_dir.exists():
            if not self.save_ckpt_dir.is_dir():
                raise Exception("given save_ckpt_dir is not directory")
        else:
            self.save_ckpt_dir.mkdir()

        avgmeter = MultiAverageMeter(self.logcols[1:])

        pbar_epoch = tqdm(
            range(self.cur_epoch, self.max_epochs),
            desc=self.stringlogger.write({"epoch": 0}),
        )
        for epoch in pbar_epoch:
            avgmeter.reset()
            # forward
            self.cur_epoch = epoch

            for idx, (real_A, real_B) in tqdm(
                enumerate(self.train_dataloader), desc="Batch", leave=False
            ):
                self.D_A.eval()
                self.D_B.eval()
                self.D_A.requires_grad_(False)
                self.D_B.requires_grad_(False)
                self.optim_G.zero_grad()

                if self.cuda:
                    real_A = real_A.cuda()
                    real_B = real_B.cuda()

                with torch.autocast(self.dev):
                    self.fakes_B = self.G_A(real_A)
                    self.rec_A = self.G_B(self.fakes_B)
                    self.fakes_A = self.G_B(real_B)
                    self.rec_B = self.G_A(self.fakes_A)

                    # optimize generators
                    loss_G = self.cyclegan_generator_loss_func(
                        real_A,
                        real_B,
                        self.fakes_A,
                        self.fakes_B,
                        self.rec_A,
                        self.rec_B,
                        self.G_A,
                        self.G_B,
                        self.D_A,
                        self.D_B,
                    )

                loss_G.backward()
                self.optim_G.step()

                # optimize discriminators
                self.D_A.train()
                self.D_B.train()
                self.D_A.requires_grad_(True)
                self.D_B.requires_grad_(True)
                self.optim_D.zero_grad()

                with torch.autocast(self.dev):
                    loss_D_A = self.cyclegan_discriminator_loss_func(
                        self.D_A, real_B, self.fakes_B
                    )
                loss_D_A.backward()

                with torch.autocast(self.dev):
                    loss_D_B = self.cyclegan_discriminator_loss_func(
                        self.D_B, real_A, self.fakes_A
                    )

                loss_D_B.backward()
                self.optim_D.step()

                losses = {
                    "loss_G": loss_G.detach().cpu().item(),
                    "loss_D_A": loss_D_A.detach().cpu().item(),
                    "loss_D_B": loss_D_B.detach().cpu().item(),
                }
                avgmeter.update(losses)

            # step lr
            self.lr_sched_G.step()
            self.lr_sched_D.step()

            # log
            vals = avgmeter.avgs()
            vals["epoch"] = epoch

            pbar_epoch.set_description(self.stringlogger.write(vals))

            if (((epoch + 1) % self.verbose) == 0) or (
                epoch + 1 in [1, self.max_epochs]
            ):
                self.csvlogger.write(vals)

            # save checkpoint
            if (((epoch + 1) % self.save_ckpt_interval) == 0) or (
                epoch + 1 == self.max_epochs
            ):
                self.save_checkpoint(self.save_ckpt_dir / f"epoch={epoch}_ckpt.pt")

            # save model
            if epoch + 1 == self.max_epochs:
                dir: Path = self.save_ckpt_dir / "results"
                if not dir.exists():
                    dir.mkdir()
                self.save_models(dir)

    def save_models(self, dir: os.PathLike):
        torch.save(self.G_A, dir / "G_A.pt")
        torch.save(self.G_B, dir / "G_B.pt")
        torch.save(self.D_A, dir / "D_A.pt")
        torch.save(self.D_B, dir / "D_B.pt")

    def save_checkpoint(self, path: os.PathLike):
        state_dict = {
            "G_A_state_dict": self.G_A.state_dict(),
            "G_B_state_dict": self.G_B.state_dict(),
            "D_A_state_dict": self.D_A.state_dict(),
            "D_B_state_dict": self.D_B.state_dict(),
            "optim_G_state_dict": self.optim_G.state_dict(),
            "optim_D_state_dict": self.optim_D.state_dict(),
            "lr_sched_G_state_dict": self.lr_sched_G.state_dict(),
            "lr_sched_D_state_dict": self.lr_sched_D.state_dict(),
            "max_epoch": self.max_epochs,
            "in_channels": self.in_channels,
            "hidden": self.hidden,
            "n_blocks": self.n_blocks,
            "n_layers": self.n_layers,
            "lr": self.lr,
            "b1": self.b1,
            "img_size": self.img_size,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "lambdaI": self.lambdaI,
            "max_epochs": self.max_epochs,
            "cur_epoch": self.cur_epoch,
        }

        torch.save(state_dict, path)
