import os
import sys
import torch
import numpy as np

from PIL import Image

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("lama")

class lama():
    def __init__(self):
        super().__init__()
        self.hub_repo = '.'
        self._model = None

    def load_model(self):
        return torch.hub.load(self.hub_repo, 'lama', source='local')

    def predict(self, img, mask):
        alpha = None
        log.debug("Image: {}".format(img.shape))
        if img.shape[2] > 3:
            log.debug("Extract alpha channel")
            alpha = img[:,:,3]
            img = img[:,:,0:3]
        log.debug("Mask: {}".format(mask.shape))
        if len(mask.shape) > 2:
            mask = np.mean(mask[:,:,0:3],2)
        output = self.model(img, mask)
        if not alpha is None:
            log.debug("Re-apply saved alpha channel")
            alpha = np.expand_dims(alpha, 2)
            output = np.concatenate((output, alpha), axis=2)
        log.debug("Output: {}".format(output.shape))
        return output

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_model()
        return self._model

def inpaintFile(inFilename, maskFilename, outFilename):
    with open(inFilename, 'rb') as file:
        img = np.asarray(Image.open(file))
    with open(maskFilename, 'rb') as file:
        mask = np.asarray(Image.open(file))
    model = lama()
    img = Image.fromarray(model.predict(img, mask))
    img.save(outFilename)
    

if __name__ == '__main__':
    if len(sys.argv) < 4 or not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
        print("Usage: {} <infilename> <maskfilename> <outfilename>".format(sys.argv[0]))
        sys.exit(-1)
    inpaintFile(sys.argv[1], sys.argv[2], sys.argv[3])

