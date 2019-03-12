from torch.utils.data import Dataset
import numpy as np


class DailyTimeSeriesFromPandas(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.list_days = self.x.index.map(lambda t: t.date()).unique() # Get list of unique days

    def __getitem__(self, item):
        day = self.list_days[item]

        x_t = self.x.loc[self.x.index.date == day]
        y_t = self.y.loc[x_t.index]
        if len(y_t)!=len(x_t):
            raise ValueError("Something wrong! Input variables have a length of %i "
                             "while output variables have a length of %i"% (len(x_t), len(y_t)))

        eps = 1e-7

        x = x_t.values.astype(np.float32)
        x -= np.min(x, axis=0)
        x /= (np.max(x, axis=0)+eps)
        x = 2*x-1
        return x, y_t.values.astype(np.float32)

    def __len__(self):
        return len(self.list_days)