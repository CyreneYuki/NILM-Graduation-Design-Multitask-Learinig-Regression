import argparse
import torch
from run_model import test_model_ae
import warnings
import matplotlib.pyplot as plt
import numpy as np
from dataloador import prompt_learn_load_data
import torch.nn as nn

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='regression', help="classification/regression")
parser.add_argument("--dataset", type=str, default='redd', help="redd/ukdale")
parser.add_argument('--save_dir', type=str, default='./models_save', help='The directory to save the trained models')
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument('--sample_rate', type=int, default=256, help='The sample interval of trainset')
parser.add_argument('--window_len', type=int, default=1024, help='The len of window, microwave=256')
parser.add_argument('--test_house', type=int, default=3, help='The num of house to test')
parser.add_argument('--applist', type=list, default=['fridge', 'dishwasher', 'microwave', 'washingmachine'],
                    help='washingmachine, fridge, dishwasher, microwave')

opt = parser.parse_args()
print(opt)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_sae(target, prediction, sample_second=6):

    r = torch.sum(target * sample_second * 1.0 / 3600.0)
    r_hat = torch.sum(prediction * sample_second * 1.0 / 3600.0)

    sae = torch.abs(r - r_hat) / torch.abs(r)

    return sae

def test(t1, t2):

    'loading data and model'
    data_path = './dataset/'+opt.dataset+'/processed/house_'
    encoder = torch.load(opt.save_dir + '/'+opt.dataset+ '/'+'/Encoder.pth').to(device)
    decoder = torch.load(opt.save_dir + '/'+opt.dataset+ '/'+'/Decoder.pth').to(device)

    AE_model = nn.Sequential(encoder, decoder)

    'testing'
    test_iterator = prompt_learn_load_data(data_path=data_path,
                                           window_len=opt.window_len,
                                           house_select=opt.test_house,
                                           applist=opt.applist,
                                           batch_size=opt.batch_size,
                                           sample_rate=opt.sample_rate,)

    agg, truth_power, pred, truth_onoff = test_model_ae(model=AE_model,
                                                  iterator=test_iterator,
                                                  device=device,)

    mean_mae = torch.mean(torch.abs(truth_power - pred))
    mean_sae = get_sae(truth_power, pred)
    print('mean mae:%.2f'%mean_mae)
    print('mean sae:%.2f'%mean_sae)

    'visualization'
    for i in range(len(opt.applist)):
        app = opt.applist[i]
        mae = torch.mean(torch.abs(truth_power[:,i] - pred[:,i]))
        sae = get_sae(truth_power[:,i], pred[:,i])
        print('-----------------')
        print('%s mae:%.2f'%(app,mae))
        print('%s sae:%.2f'%(app,sae))

        x_axis = np.linspace(0, len(agg[t1:t2]), len(agg[t1:t2])).astype(int)

        plt.plot(x_axis, agg[t1:t2])
        plt.plot(x_axis, truth_power[t1:t2,i])
        plt.plot(x_axis, pred[t1:t2,i])
        plt.legend(['Agg power', 'Truth '+app+' Power', 'Pred '+app+' Power'])

        plt.show()


test(0, 20000)