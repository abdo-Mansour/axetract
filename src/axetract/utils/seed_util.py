import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    '''
    Set the random seed for reproducibility across various libraries.
    Args:
        seed (int): The seed value to set. Default is 42.

    '''
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

