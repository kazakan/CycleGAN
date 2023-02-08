import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from cyclegan.dataset.img_folder import ImageFolder


def main(args):
    cuda = args.cuda
    dev = "cpu" if cuda else "cuda"

    model = torch.load(args.model)

    model.to(dev)
    model.eval()

    dataset = ImageFolder(args.source, transform=T.ToTensor())
    dataloader = DataLoader(dataset)

    toimage = T.ToPILImage()
    imgs = []

    with torch.no_grad():
        for idx, img in tqdm(
            enumerate(dataloader), desc="Processing", total=len(dataloader)
        ):
            if cuda:
                img = img.cuda()

            converted_img = model(img)
            converted_img = converted_img.detach().cpu()
            converted_img = converted_img.squeeze()
            converted_img = toimage(converted_img)
            imgs.append(converted_img)

    for idx, img in enumerate(imgs):
        img.save(args.dest / dataset.imgs[idx].name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)

    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        help="Use cuda or not",
    )

    args = parser.parse_args()
    main(args)
