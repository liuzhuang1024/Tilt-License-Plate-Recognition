import torch.nn as nn
from models.morn import MORN
from models.asrn_res import ASRN
from models.stn import STN


class MORAN(nn.Module):

    def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False,
                 inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):
        super(MORAN, self).__init__()
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA)
        self.ASRN = ASRN(targetH, nc, nclass, nh, BidirDecoder, CUDA)
        self.STN = STN()

    def forward(self, x, length, text, text_rev, test=False, debug=False):
        if debug:
            # x_rectified, demo = self.MORN(x, test, debug=debug)
            x_rectified, demo = self.STN(x, test)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds, demo
        else:
            # x_rectified = self.MORN(x, test, debug=debug)
            x_rectified = self.MORN(x, test)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds
