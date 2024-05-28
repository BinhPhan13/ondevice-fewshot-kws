import torch

MODEL_REGISTRY = {}

def register_model(model_name):
    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator

def get_model(model_opt):
    model_name = model_opt['model_name']
    del model_opt['model_name']
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](**model_opt)
    else:
        raise ValueError("Unknown model {:s}".format(model_name))

        
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
#     print(f'test: {torch.pow(x - y, 2).sum(2).size()}')
    return torch.pow(x - y, 2).sum(2)



def mahalanobis_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
#     print(f'x: {x.size()}')
#     print(f'y: {y.size()}')
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
#     print(f'x: {x.size()}')
#     print(f'y: {y.size()}')
    dist = torch.norm(x-y, p = 2, dim = 2)
#     print(dist.size()); exit(0)
    return dist