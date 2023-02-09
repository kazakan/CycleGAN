class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiAverageMeter:
    def __init__(self, names):
        self.names = names
        self._counters = {}
        for n in names:
            self._counters[n] = AverageMeter()

    def __getitem__(self, key):
        if key not in self.names:
            raise KeyError()

        return self._counters[key]

    def update(self, val):
        if type(val) is not dict:
            raise ValueError("val should be dictionary")

        for k, v in val.items():
            if k in self.names:
                self[k].update(v)

    def reset(self):
        for n in self.names:
            self._counters[n].reset()

    def avgs(self):
        ret = {}
        for n in self.names:
            ret[n] = self._counters[n].avg
        return ret
