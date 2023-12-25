import torch.utils.data as data
import pandas as pd
import numpy as np


class PromptLearnGenerator(data.Dataset):
    def __init__(self, agg=None, app=None, app_onoff=None, window_len=1024, sample_rate=16):
        self.agg = agg
        self.app = app
        self.app_onoff = app_onoff
        self.window_len = window_len
        self.sample_rate = sample_rate
        self.data_len = (len(self.agg) - self.window_len) // self.sample_rate - 1

    def __getitem__(self, item):

        "得到片段起止点"
        start = item % self.data_len
        s1 = start * self.sample_rate
        s2 = s1 + self.window_len

        "截取片段"
        agg = self.agg[s1:s2]
        app = self.app[s1:s2]
        app_onoff = self.app_onoff[s1:s2]

        return agg, app, app_onoff

    def __len__(self):
        return self.data_len


def prompt_learn_load_data(data_path, window_len, house_select, applist, batch_size=32, sample_rate=16, shuffle=False):

    x_path = data_path + str(house_select) + '.csv'
    data_frame = pd.read_csv(x_path, header=0)

    x_onoff_path = data_path + str(house_select) + '_onoff.csv'
    data_onoff_frame = pd.read_csv(x_onoff_path, header=0)

    agg_data = data_frame['mains'].to_numpy(np.float32)
    app_data = data_frame[applist].to_numpy(np.float32)
    app_onoff_data = data_onoff_frame[applist].to_numpy(np.float32)

    data_loader = PromptLearnGenerator(agg=agg_data,
                                       app=app_data,
                                       app_onoff=app_onoff_data,
                                       window_len=window_len,
                                       sample_rate=sample_rate)

    data_iterator = data.DataLoader(data_loader, batch_size=batch_size, shuffle=shuffle)

    return data_iterator

