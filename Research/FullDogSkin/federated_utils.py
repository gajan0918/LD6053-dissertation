import torch

def get_params(model):
    return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

def set_params(model, params, device):
    sd = {k: params[k].to(device) for k in params}
    model.load_state_dict(sd)

def average_params(client_params, weights):
    avg = {}
    for k in client_params[0].keys():
        avg[k] = sum(client_params[i][k] * weights[i] for i in range(len(client_params)))
    return avg
