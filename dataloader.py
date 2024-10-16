import torch.utils.data as data
import torch


class TrainDataset(data.Dataset):
    def __init__(self, train_list):
        self.train_list = train_list
        self.init()

    def init(self):
        self.x = []
        self.y = []
        self.p = []
        with open(self.train_list) as f:
            for line in f.readlines():
                data = line.strip().split(' ')
                self.x.append(data[0])
                if len(data) > 1:
                    self.y.append(data[1])
                    if len(data) > 2:
                        self.p.append(data[2])
                    else:
                        self.p.append(-1)
                else:
                    self.y.append(-1)


    def __getitem__(self, index):
        if index < 0 or index >= len(self.x):
            print("Index out of range")
        x = float(self.x[index])
        y = float(self.y[index])
        p = float(self.p[index])
        tensor_xy = torch.tensor([x, y])
        p = torch.tensor([p])

        return tensor_xy, p

    def __len__(self):
        return len(self.x)

def data_loader(args):
    dataset = TrainDataset(
        args.train_list
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_loader
