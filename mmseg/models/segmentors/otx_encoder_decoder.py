# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmseg.models import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class OTXEncoderDecoder(EncoderDecoder):
    def simple_test(self, img, img_meta, rescale=True, output_logits=False):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if output_logits:
            seg_pred = seg_logit
        else:
            if self.out_channels == 1:
                seg_pred = (seg_logit > self.decode_head.threshold).to(seg_logit).squeeze(1)
            else:
                seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            if seg_pred.dim() != 4:
                seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred
