import argparse
import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from dataloador import prompt_learn_load_data
from run_model import run_model_ae
from model import AggEncoder, AggDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='redd', help="dataset select")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument('--window_len', type=int, default=1024, help='The len of window, microwave=256')
parser.add_argument('--train_house', type=int, default=1, help='Which house is used to train')
parser.add_argument('--val_house', type=int, default=3, help='Which house is used to validationy')
parser.add_argument('--save_dir', type=str, default='./models_save', help='The directory to save the trained models')
parser.add_argument('--sample_rate', type=int, default=256, help='The sample interval of trainset')
parser.add_argument('--applist', type=list, default=['fridge', 'dishwasher', 'microwave', 'washingmachine'],
                    help='washingmachine, fridge, dishwasher, microwave')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = './dataset/'+opt.dataset+'/processed/house_'

encoder = AggEncoder().to(device)
decoder = AggDecoder().to(device)

ae = nn.Sequential(encoder, decoder)

'optimizer'
optimizer_AE = torch.optim.Adam(ae.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

'loss function'
ae_loss = torch.nn.MSELoss().to(device)

'data iterator'
train_iterator = prompt_learn_load_data(data_path,
                                        window_len=opt.window_len,
                                        house_select=opt.train_house,
                                        applist=opt.applist,
                                        batch_size=opt.batch_size,
                                        sample_rate=opt.sample_rate,
                                        shuffle=True,
                                        )

val_iterator = prompt_learn_load_data(data_path,
                                      window_len=opt.window_len,
                                      house_select=opt.val_house,
                                      applist=opt.applist,
                                      batch_size=opt.batch_size,
                                      sample_rate=opt.window_len,)

'Training'
loss_list = []
best_loss = 100000

for epoch in range(opt.n_epochs):

    train_loss = run_model_ae(model=ae,
                              iterator=train_iterator,
                              optimizer=optimizer_AE,
                              criterion=ae_loss,
                              device=device,
                              if_train=True)

    val_loss = run_model_ae(model=ae,
                            iterator=val_iterator,
                            optimizer=optimizer_AE,
                            criterion=ae_loss,
                            device=device,
                            if_train=False)

    loss_list.append(val_loss)
    if val_loss <= best_loss:
        print('\t\tthe best val loss gotten. saving......')
        best_loss = val_loss
        best_epoch = epoch
        if not os.path.exists(opt.save_dir + '/'+opt.dataset):  # 判断当前路径是否存在，没有则创建new文件夹
            os.makedirs(opt.save_dir + '/'+opt.dataset)
        torch.save(encoder, opt.save_dir + '/'+opt.dataset+'/Encoder.pth')
        torch.save(decoder, opt.save_dir + '/'+opt.dataset+'/Decoder.pth')

    print(f'Epoch: {epoch}\tTrain_loss: {train_loss}\tVal_loss: {val_loss}')

'draw loss'
plt.plot(loss_list)
plt.show()
