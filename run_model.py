import torch
from tqdm import tqdm


def run_model_ae(model, iterator, optimizer, criterion, device, if_train=True):
    model.eval()
    if if_train:
        model.train()
    progress_bar = tqdm(iterator)
    loss_list = []
    for i, (inp, app, app_onoff) in enumerate(progress_bar):
        x = inp.to(device)
        y = app.to(device)
        optimizer.zero_grad()

        rebuild = model(x)
        loss = criterion(rebuild, y)
        loss.backward()
        optimizer.step()

        loss_list.append(loss)

    return torch.mean(torch.Tensor(loss_list))


def test_model_ae(model, iterator, device):
    model.eval()
    window = iterator.dataset.window_len
    sample_rate = iterator.dataset.sample_rate

    progress_bar = tqdm(iterator)
    agg_list, truth_list, pred_list, truth_onoff_list = [], [], [], []

    with torch.no_grad():
        for i, (agg, app, app_onoff) in enumerate(progress_bar):
            x = agg.to(device)
            truth = app.to(device)
            truth_onoff = app_onoff.to(device)

            pred = model(x)

            start = (window - sample_rate) // 2

            agg_slice = agg[:, start : start+sample_rate]
            truth_slice = truth[:, start : start+sample_rate]
            pred_slice = pred[:, start : start+sample_rate]
            truth_onoff_slice = truth_onoff[:, start : start+sample_rate]

            agg_list.append(agg_slice.reshape(-1, 1))
            truth_list.append(truth_slice.reshape(-1, 4))
            pred_list.append(pred_slice.reshape(-1, 4))
            truth_onoff_list.append(truth_onoff_slice.reshape(-1, 4))

        truth_list = torch.cat(truth_list).squeeze().to('cpu')
        agg_list = torch.cat(agg_list).squeeze().to('cpu')
        pred_list = torch.cat(pred_list).squeeze().to('cpu')
        truth_onoff_list = torch.cat(truth_onoff_list).squeeze().to('cpu')

        return agg_list, truth_list, pred_list, truth_onoff_list
