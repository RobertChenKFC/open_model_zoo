"""
Copyright (c) 2018-2021 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
from .adapter import Adapter
from ..representation import SalientRegionPrediction
from ..config import StringField


class SalientObjectDetection(Adapter):
    __provider__ = 'salient_object_detection'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'salient_map_output': StringField(optional=True, description='target output layer for getting salience map')
        })
        return params

    def configure(self):
        self.salient_map_output = self.get_value_from_config('salient_map_output')

    def process(self, raw, identifiers, frame_meta):
        raw_output = self._extract_predictions(raw, frame_meta)
        if self.salient_map_output is None:
            self.select_output_blob(raw_output)
            self.salient_map_output = self.output_blob
        result = []
        for identifier, mask in zip(identifiers, raw_output[self.salient_map_output]):
            # For some reason, the original code applied the sigmoid function
            # to the mask before using it (maybe the model didn't contain the
            # sigmoid layer). We removed the code.
            #
            # Also, the original code rounded the mask, effectively doing the
            # thresholding (of 0.5, since it used np.round) and returning a
            # binary map. We eliminate that so that the original floating
            # point probability map is returned.
            #
            # Note that, however, that the prediction map received when
            # calculating the metrics is not the floating point map returned
            # here, but rather the quantized (to 0-255) version of this map.
            # (check out ResizeSegmentationMask._bytescale in
            # resize_segmentation_mask.py for more detail). Therefore,
            # we change the annotation mask so that it also returns a 0-255 map
            # (check out SalientMapNormalizer.process_image in
            # normalize_salient_map.py for more detail).
            result.append(SalientRegionPrediction(identifier, np.squeeze(mask)))
        return result
