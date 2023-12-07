dependencies = [
    'torch', 'numpy', 
    'omegaconf', 'albumentations', 'webdataset', 'tqdm', 'easydict', 
    'sklearn', 'pandas', 'torchvision', 'pytorch_lightning', 'kornia',
]

import os
import requests
import tempfile
import zipfile
import shutil
import yaml
import numpy as np
import torch.hub

from omegaconf import OmegaConf
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

import logging

#logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("lama")

class _lama():

    def __init__(self, model, key, device):
        self._model = model
        self._key = key
        self._device = device

    def __call__(self, image, mask):

        # image: (h,w,c) [0..255]
        # mask: (h,w) [0..255]

        assert len(mask.shape) == 2

        log.debug("Input: {}".format(image.shape))
        log.debug("Mask: {}".format(mask.shape))
        
        image = torch.from_numpy(image.copy()).float().div(255)
        mask = torch.from_numpy(mask.copy()).float().div(255)

        mod = 8
        unpad_to_size = image.shape[0:2]

        batch = {}

        # (h,w,c) -> (b,c,h,w)
        batch['image'] = image.permute(2, 0, 1).unsqueeze(0)
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        
        # (h,w)
        batch['mask'] = mask[None, None]
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        # 0: retain, 1: masked/inpaint
        batch['mask'] = (batch['mask'] < 1) * 1

        batch = move_to_device(batch, self._device)
        batch = self._model(batch)

        # (c,h,w) -> (h,w,c)
        cur_res = batch[self._key][0]
        cur_res = cur_res.permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        output = np.clip(cur_res, 0.0, 1.0)
        output = (output * 255.0).astype(np.uint8)

        log.debug("Output: {}".format(output.shape))
        return output

def lama(progress=True, map_location=None, allow_mps=False):

    if (map_location == torch.device('mps')) and not allow_mps:
        map_location = torch.device('cpu')

    MODEL_URL = 'https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip'

    TARGET = os.path.join(torch.hub.get_dir(), "lama")
    MODEL = "big-lama"

    if not os.path.isdir(os.path.join(TARGET, MODEL)):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "{}.zip".format(MODEL))
            print("Downloading {} to {}".format(MODEL_URL, path))
            torch.hub.download_url_to_file(MODEL_URL, path, progress=progress)
            with zipfile.ZipFile(path, 'r') as zip:
                zip.extractall(tmp)
            os.makedirs(TARGET, exist_ok=True)
            shutil.move(os.path.join(tmp, MODEL), TARGET)

    train_config_path = os.path.join(TARGET, MODEL, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(TARGET, MODEL, "models", "best.ckpt")
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(map_location)

    model = _lama(model, 'inpainted', map_location)
    return model
