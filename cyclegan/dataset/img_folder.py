import os
from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision.datasets.folder import default_loader, is_image_file


def to_path(path: os.PathLike):
    return Path(path)


def make_dataset(dir, max_len=float("inf")):
    dir = to_path(dir)
    images = []

    if not dir.is_dir():
        raise Exception("dir should be directory.")

    images = [p for p in dir.glob("./*") if is_image_file(str(p))]

    return images[: min(max_len, len(images))]


class ImageFolder(Dataset):
    def __init__(
        self, root, transform=None, loader=default_loader, max_len=float("inf")
    ):
        super().__init__()
        imgs = make_dataset(root, max_len=max_len)
        if len(imgs) == 0:
            raise Exception(f"0 images in {root}")

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            return self.transform(img)
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        root1,
        root2,
        transform=None,
        loader=default_loader,
        # unmatch_length_option="squeeze",
    ) -> None:
        super().__init__()

        self.set1 = ImageFolder(root1, transform, loader)
        self.set2 = ImageFolder(root2, transform, loader)

        # match size
        if len(self.set1) > len(self.set2):
            self.set1 = Subset(self.set1, list(range(len(self.set2))))
        elif len(self.set1) < len(self.set2):
            self.set2 = Subset(self.set2, list(range(len(self.set1))))

    def __getitem__(self, index):
        return self.set1[index], self.set2[index]

    def __len__(self):
        return len(self.set1)
