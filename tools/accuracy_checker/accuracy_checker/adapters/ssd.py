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

import itertools
import math
import re
from typing import Tuple

import numpy as np
import numba as nb

from .adapter import Adapter
from ..config import ConfigValidator, StringField, NumberField, ListField, BoolField
from ..postprocessor import NMS
from ..representation import DetectionPrediction, ContainerPrediction


class SSDAdapter(Adapter):
    """
    Class for converting output of SSD model to DetectionPrediction representation
    """
    __provider__ = 'ssd'
    prediction_types = (DetectionPrediction, )

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        prediction_batch = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(prediction_batch)
        prediction_batch = prediction_batch[self.output_blob]
        prediction_batch = prediction_batch.reshape(-1, 7)
        prediction_batch = self.remove_empty_detections(prediction_batch)

        result = []
        for batch_index, identifier in enumerate(identifiers):
            prediction_mask = np.where(prediction_batch[:, 0] == batch_index)
            detections = prediction_batch[prediction_mask]
            detections = detections[:, 1::]
            result.append(DetectionPrediction(identifier, *zip(*detections)))

        return result

    @staticmethod
    def remove_empty_detections(prediction_blob):
        ind = prediction_blob[:, 0]
        ind_ = np.where(ind == -1)[0]
        m = ind_[0] if ind_.size else prediction_blob.shape[0]

        return prediction_blob[:m, :]


class PyTorchSSDDecoder(Adapter):
    """
    Class for converting output of PyTorch SSD models to DetectionPrediction representation
    """
    __provider__ = 'pytorch_ssd_decoder'

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'scores_out': StringField(description="Scores output layer name."),
            'boxes_out': StringField(description="Boxes output layer name."),
            'confidence_threshold': NumberField(optional=True, default=0.05, description="Confidence threshold."),
            'nms_threshold': NumberField(optional=True, default=0.5, description="NMS threshold."),
            'keep_top_k': NumberField(optional=True, value_type=int, default=200, description="Keep top K."),
            'feat_size': ListField(
                optional=True, description='Feature sizes list',
                value_type=ListField(value_type=NumberField(min_value=1, value_type=int))
            ),
            'do_softmax': BoolField(
                optional=True, default=True, description='Softmax operation should be applied to scores or not'
            )
        })

        return parameters

    def configure(self):
        self.scores_out = self.get_value_from_config('scores_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.confidence_threshold = self.get_value_from_config('confidence_threshold')
        self.nms_threshold = self.get_value_from_config('nms_threshold')
        self.keep_top_k = self.get_value_from_config('keep_top_k')
        self.do_softmax = self.get_value_from_config('do_softmax')
        feat_size = self.get_value_from_config('feat_size')

        # Set default values according to:
        # https://github.com/mlcommons/inference/tree/master/cloud/single_stage_detector
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]] if feat_size is None else feat_size
        self.scales = [21, 45, 99, 153, 207, 261, 315]
        self.strides = [3, 3, 2, 2, 2, 2]
        self.scale_xy = 0.1
        self.scale_wh = 0.2

    @staticmethod
    def softmax(x, axis=0):
        return np.transpose(np.transpose(np.exp(x)) * np.reciprocal(np.sum(np.exp(x), axis=axis)))

    @staticmethod
    def default_boxes(fig_size, feat_size, scales, aspect_ratios):

        fig_size_w, fig_size_h = fig_size
        scales = [(int(s * fig_size_w / 300), int(s * fig_size_h / 300)) for s in scales]
        fkw, fkh = np.transpose(feat_size)

        default_boxes = []
        for idx, sfeat in enumerate(feat_size):
            sfeat_w, sfeat_h = sfeat
            sk1 = scales[idx][0] / fig_size_w
            sk2 = scales[idx + 1][1] / fig_size_h
            sk3 = math.sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat_w), range(sfeat_h)):
                    cx, cy = (j + 0.5) / fkh[idx], (i + 0.5) / fkw[idx]
                    default_boxes.append((cx, cy, w, h))
        default_boxes = np.clip(default_boxes, 0, 1)

        return default_boxes

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """

        raw_outputs = self._extract_predictions(raw, frame_meta)

        batch_scores = raw_outputs[self.scores_out]
        batch_boxes = raw_outputs[self.boxes_out]
        need_transpose = np.shape(batch_boxes)[-1] != 4

        result = []
        for identifier, scores, boxes, meta in zip(identifiers, batch_scores, batch_boxes, frame_meta):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            image_info = meta.get("image_info")[0:2]

            # Default boxes
            dboxes = self.default_boxes(image_info, self.feat_size, self.scales, self.aspect_ratios)

            # Scores
            scores = np.transpose(scores) if need_transpose else scores
            if self.do_softmax:
                scores = self.softmax(scores, axis=1)

            # Boxes
            boxes = np.transpose(boxes) if need_transpose else boxes
            boxes[:, :2] = self.scale_xy * boxes[:, :2]
            boxes[:, 2:] = self.scale_wh * boxes[:, 2:]
            boxes[:, :2] = boxes[:, :2] * dboxes[:, 2:] + dboxes[:, :2]
            boxes[:, 2:] = np.exp(boxes[:, 2:]) * dboxes[:, 2:]

            for label, score in enumerate(np.transpose(scores)):

                # Skip background label
                if label == 0:
                    continue

                # Filter out detections with score < confidence_threshold
                mask = score > self.confidence_threshold
                filtered_boxes, filtered_score = boxes[mask, :], score[mask]
                if filtered_score.size == 0:
                    continue

                # Transform to format (x_min, y_min, x_max, y_max)
                x_mins = (filtered_boxes[:, 0] - 0.5 * filtered_boxes[:, 2])
                y_mins = (filtered_boxes[:, 1] - 0.5 * filtered_boxes[:, 3])
                x_maxs = (filtered_boxes[:, 0] + 0.5 * filtered_boxes[:, 2])
                y_maxs = (filtered_boxes[:, 1] + 0.5 * filtered_boxes[:, 3])

                # Apply NMS
                keep = NMS.nms(x_mins, y_mins, x_maxs, y_maxs, filtered_score, self.nms_threshold,
                               include_boundaries=False, keep_top_k=self.keep_top_k)

                filtered_score = filtered_score[keep]
                x_mins = x_mins[keep]
                y_mins = y_mins[keep]
                x_maxs = x_maxs[keep]
                y_maxs = y_maxs[keep]

                # Keep topK
                # Applied just after NMS - no additional sorting is required for filtered_score array
                filtered_score = filtered_score[:self.keep_top_k]
                x_mins = x_mins[:self.keep_top_k]
                y_mins = y_mins[:self.keep_top_k]
                x_maxs = x_maxs[:self.keep_top_k]
                y_maxs = y_maxs[:self.keep_top_k]

                # Save detections
                labels = np.full_like(filtered_score, label)
                detections['labels'].extend(labels)
                detections['scores'].extend(filtered_score)
                detections['x_mins'].extend(x_mins)
                detections['y_mins'].extend(y_mins)
                detections['x_maxs'].extend(x_maxs)
                detections['y_maxs'].extend(y_maxs)

            result.append(
                DetectionPrediction(
                    identifier, detections['labels'], detections['scores'], detections['x_mins'],
                    detections['y_mins'], detections['x_maxs'], detections['y_maxs']
                )
            )

        return result


class FacePersonAdapter(Adapter):
    __provider__ = 'face_person_detection'
    prediction_types = (DetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'face_out': StringField(description="Face detection output layer name."),
            'person_out': StringField(description="Person detection output layer name"),
        })

        return parameters

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )

    def configure(self):
        self.face_detection_out = self.launcher_config['face_out']
        self.person_detection_out = self.launcher_config['person_out']
        self.face_adapter = SSDAdapter(self.launcher_config, self.label_map, self.face_detection_out)
        self.person_adapter = SSDAdapter(self.launcher_config, self.label_map, self.person_detection_out)

    def process(self, raw, identifiers, frame_meta):
        face_batch_result = self.face_adapter.process(raw, identifiers, frame_meta)
        person_batch_result = self.person_adapter.process(raw, identifiers, frame_meta)
        result = [ContainerPrediction({self.face_detection_out: face_result, self.person_detection_out: person_result})
                  for face_result, person_result in zip(face_batch_result, person_batch_result)]

        return result


class SSDAdapterMxNet(Adapter):
    """
    Class for converting output of MXNet SSD model to DetectionPrediction representation
    """
    __provider__ = 'ssd_mxnet'

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model which is ndarray of shape (batch, det_count, 6),
                 each detection is defined by 6 values: class_id, prob, x_min, y_min, x_max, y_max
        Returns:
            list of DetectionPrediction objects
        """
        raw_outputs = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_outputs)
        result = []
        for identifier, prediction_batch in zip(identifiers, raw_outputs[self.output_blob]):
            # Filter detections (get only detections with class_id >= 0)
            detections = prediction_batch[np.where(prediction_batch[:, 0] >= 0)]
            # Append detections to results
            result.append(DetectionPrediction(identifier, *zip(*detections)))

        return result


class SSDONNXAdapter(Adapter):
    __provider__ = 'ssd_onnx'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'labels_out': StringField(description='name (or regex for it) of output layer with labels'),
                'scores_out': StringField(
                    description='name (or regex for it) of output layer with scores', optional=True
                ),
                'bboxes_out': StringField(description='name (or regex for it) of output layer with bboxes')
            }
        )
        return parameters

    def configure(self):
        self.labels_out = self.get_value_from_config('labels_out')
        self.scores_out = self.get_value_from_config('scores_out')
        self.bboxes_out = self.get_value_from_config('bboxes_out')
        self.outputs_verified = False

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        results = []
        if not self.outputs_verified:
            self._get_output_names(raw_outputs)
        boxes_out, labels_out = raw_outputs[self.bboxes_out], raw_outputs[self.labels_out]
        if len(boxes_out.shape) == 2:
            boxes_out = np.expand_dims(boxes_out, 0)
            labels_out = np.expand_dims(labels_out, 0)
        for idx, (identifier, bboxes, labels) in enumerate(zip(
                identifiers, boxes_out, labels_out
        )):
            if self.scores_out:
                scores = raw_outputs[self.scores_out][idx]
                x_mins, y_mins, x_maxs, y_maxs = bboxes.T
            else:
                x_mins, y_mins, x_maxs, y_maxs, scores = bboxes.T
            if labels.ndim > 1:
                labels = np.squeeze(labels)
            if scores.ndim > 1:
                scores = np.squeeze(scores)
            if x_mins.ndim > 1:
                x_mins = np.squeeze(x_mins)
                y_mins = np.squeeze(y_mins)
                x_maxs = np.squeeze(x_maxs)
                y_maxs = np.squeeze(y_maxs)
            results.append(
                DetectionPrediction(
                    identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return results

    def _get_output_names(self, raw_outputs):
        labels_regex = re.compile(self.labels_out)
        scores_regex = re.compile(self.scores_out) if self.scores_out else ''
        bboxes_regex = re.compile(self.bboxes_out)

        def find_layer(regex, output_name, all_outputs):
            suitable_layers = [layer_name for layer_name in all_outputs if regex.match(layer_name)]
            if not suitable_layers:
                raise ValueError('suitable layer for {} output is not found'.format(output_name))

            if len(suitable_layers) > 1:
                raise ValueError('more than 1 layers matched to regular expression, please specify more detailed regex')

            return suitable_layers[0]

        self.labels_out = find_layer(labels_regex, 'labels', raw_outputs)
        self.scores_out = find_layer(scores_regex, 'scores', raw_outputs) if self.scores_out else None
        self.bboxes_out = find_layer(bboxes_regex, 'bboxes', raw_outputs)

        self.outputs_verified = True


class SSDFaceDetectionAdapter(Adapter):
    __provider__ = "ssd_face_detection"

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            "box_offsets_out": StringField(
                description="name (or regex for it) of output layer with "
                            "bounding box offsets."
            ),
            "class_predictions_out": StringField(
                description="name (or regex for it) of output layer with "
                            "class predictions."
            ),
            "image_width": NumberField(
                description="width of the model image input"
            ),
            "image_height": NumberField(
                description="height of the model image input"
            )
        })
        return parameters

    def configure(self):
        self.box_offsets_out = self.get_value_from_config(
            "box_offsets_out"
        )
        self.class_predictions_out = self.get_value_from_config(
            "class_predictions_out"
        )
        self.img_width = np.int32(self.get_value_from_config("image_width"))
        self.img_height = np.int32(self.get_value_from_config("image_height"))
        self.prior_boxes = self.generate_all_prior_boxes()

    @staticmethod
    @nb.njit
    def generate_prior_boxes(
        img_width: int, img_height: int, layer_width: int, layer_height: int,
        aspect_ratios: np.array, min_size: int, max_size: int, step: int,
        offset: float
    ) -> np.array:
        """
        Generates a np array of shape (1, num_prior_boxes, 4) where each group
        of 4 values corresponds to a prior boxes' center_x, center_y, box_width,
        box_height.
        """
        # Add all the reciprocals of the aspect ratios, too
        num_aspect_ratios = len(aspect_ratios)
        all_aspect_ratios = np.zeros((2 * num_aspect_ratios,), dtype=np.float32)
        for i in range(num_aspect_ratios):
            all_aspect_ratios[i * 2] = aspect_ratios[i]
            all_aspect_ratios[i * 2 + 1] = 1. / aspect_ratios[i]

        num_aspect_ratios = len(all_aspect_ratios)
        num_prior_boxes = layer_width * layer_height * (num_aspect_ratios + 2)
        prior_boxes = np.zeros((1, num_prior_boxes, 4), dtype=np.float32)

        # Boxes:
        # 1.     width:  min_size
        #        height: min_size
        # 2.     width:  sqrt(min_size * max_size)
        #        height: sqrt(min_size * max_size)
        # i + 2. width:  min_size / sqrt(aspect_ratios[i]),
        #        height: max_size * sqrt(aspect_ratios[i])
        box_widths = np.zeros((2 + num_aspect_ratios,), dtype=np.int32)
        box_heights = np.zeros((2 + num_aspect_ratios,), dtype=np.int32)
        box_widths[:2] = box_heights[:2] = (
            min_size, int(np.sqrt(min_size * max_size))
        )
        box_widths[2:] = min_size * np.sqrt(all_aspect_ratios)
        box_heights[2:] = min_size / np.sqrt(all_aspect_ratios)

        # Fill in box coordinates in the first half of the input
        idx = 0
        for i in range(layer_height):
            for j in range(layer_width):
                center_x = (j + offset) * step
                center_y = (i + offset) * step

                for k in range(len(box_widths)):
                    box_width = box_widths[k]
                    box_height = box_heights[k]
                    prior_boxes[0, idx, :] = np.array([
                        center_x / img_width, center_y / img_height,
                        box_width / img_width, box_height / img_height
                    ], dtype=np.float32)
                    idx += 1

        return prior_boxes

    def generate_all_prior_boxes(self):
        layer_dimensions = [
            (38, 38), (19, 19), (10, 10), (5, 5), (5, 5), (5, 5)
        ]
        all_aspect_ratios = [
            [2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]
        ]
        min_max_sizes = [
            (30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)
        ]
        steps = [8, 16, 32, 64, 100, 300]
        all_prior_boxes = np.concatenate([
            self.generate_prior_boxes(
                self.img_width, self.img_height, layer_width, layer_height,
                np.array(aspect_ratio, dtype=np.float32),
                min_size, max_size, step, 0.5
            ) for (
                (layer_width, layer_height), aspect_ratio, (min_size, max_size),
                step
            ) in zip(layer_dimensions, all_aspect_ratios, min_max_sizes, steps)
        ], axis=1)
        return all_prior_boxes

    @staticmethod
    @nb.njit
    def get_bounding_boxes(
        num_classes: np.int32, box_offsets: np.array,
        class_predictions: np.array, prior_boxes: np.array,
        variances: np.array, confidence_threshold: np.float32
    ) -> Tuple[np.array, np.array]:
        """Returns two np arrays: bounding_boxes and classes. Bounding_boxes
        is of shape (1, num_prior_boxes, 5), and each group of 5 values
        corresponds to min x, min y, max x, max y, confidence. Classes is of shape
        (1, num_prior_boxes), where each value corresponds to the class index.
        """
        num_prior_boxes = prior_boxes.shape[1]
        bounding_boxes = np.zeros((1, num_prior_boxes, 5), dtype=np.float32)
        classes = np.zeros((1, num_prior_boxes), dtype=np.int32)

        (
            center_x_variance, center_y_variance,
            box_width_variance, box_height_variance
        ) = variances
        idx = 0
        for i in range(num_prior_boxes):
            max_prob_class = np.argmax(
                class_predictions[
                    0,
                    i * (num_classes + 1) + 1:
                    i * (num_classes + 1) + 1 + num_classes
                ]
            ) + i * (num_classes + 1) + 1
            max_prob = class_predictions[0, max_prob_class]
            if max_prob < confidence_threshold:
                continue
            (
                center_x_offset, center_y_offset,
                box_width_offset, box_height_offset
            ) = box_offsets[0, i * 4: (i + 1) * 4]

            center_x, center_y, box_width, box_height = prior_boxes[0, i, :]
            center_x += center_x_offset * box_width * center_x_variance
            center_y += center_y_offset * box_height * center_y_variance
            box_width *= np.exp(box_width_offset * box_width_variance) / 2
            box_height *= np.exp(box_height_offset * box_height_variance) / 2

            bounding_boxes[0, idx, :] = np.array([
                # min x
                center_x - box_width,
                # min y
                center_y - box_height,
                # max x
                center_x + box_width,
                # max y
                center_y + box_height,
                # confidence
                max_prob
            ], dtype=np.float32)
            classes[0, idx] = max_prob_class
            idx += 1

        return bounding_boxes[:, :idx, :], classes[:, :idx]

    @staticmethod
    @nb.njit
    def nms(
        bounding_boxes: np.array, classes: np.array, nms_threshold: float
    ) -> Tuple[np.array, np.array]:
        """
        Uses Non-Maximum Suppression to extract the bounding boxes and its
        corresponding class indices together.
        """
        num_boxes = bounding_boxes.shape[1]
        box_indices = np.arange(num_boxes, dtype=np.int32)

        chosen_box_indices = np.zeros((num_boxes,), dtype=np.int32)
        num_chosen_box_indices = 0
        while num_boxes != 0:
            max_index = 0
            cur_max = -1
            for i in range(num_boxes):
                i = box_indices[i]
                cur = bounding_boxes[0, i, 4]
                if cur > cur_max:
                    max_index = i
                    cur_max = cur
            chosen_box_indices[num_chosen_box_indices] = max_index
            num_chosen_box_indices += 1

            new_box_indices = np.zeros((num_boxes,), dtype=np.int32)
            new_num_boxes = 0
            min_x_1, min_y_1, max_x_1, max_y_1 = bounding_boxes[0, max_index,
                                                 :4]
            for i in range(num_boxes):
                i = box_indices[i]
                keep = False
                if i != max_index:
                    min_x_2, min_y_2, max_x_2, max_y_2 = bounding_boxes[0, i,
                                                         :4]
                    if min_x_1 > max_x_2 or min_x_2 > max_x_1:
                        overlap_x = np.float32(0)
                    elif min_x_1 <= max_x_2:
                        overlap_x = max_x_2 - max(min_x_1, min_x_2)
                    else:
                        overlap_x = max_x_1 - max(min_x_2, min_x_1)
                    if min_y_1 > max_y_2 or min_y_2 > max_y_1:
                        overlap_y = np.float32(0)
                    elif min_y_1 <= max_y_2:
                        overlap_y = max_y_2 - max(min_y_1, min_y_2)
                    else:
                        overlap_y = max_y_1 - max(min_y_2, min_y_1)
                    overlap = overlap_x * overlap_y
                    area_1 = (max_x_1 - min_x_1) * (max_y_1 - min_y_1)
                    area_2 = (max_x_2 - min_x_2) * (max_y_2 - min_y_2)
                    iou = overlap / (area_1 + area_2 - overlap)
                    keep = iou <= nms_threshold
                if keep:
                    new_box_indices[new_num_boxes] = i
                    new_num_boxes += 1
            box_indices = new_box_indices
            num_boxes = new_num_boxes

        chosen_box_indices = chosen_box_indices[:num_chosen_box_indices]
        new_bounding_boxes = np.zeros(
            (1, num_chosen_box_indices, 5), dtype=np.float32
        )
        new_classes = np.zeros(
            (1, num_chosen_box_indices), dtype=np.int32
        )
        for i in range(num_chosen_box_indices):
            j = chosen_box_indices[i]
            new_bounding_boxes[0, i, :] = bounding_boxes[0, j, :]
            new_classes[0, i] = classes[0, j]
        return new_bounding_boxes, new_classes

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        box_offsets = np.array(
            raw_outputs[self.box_offsets_out], dtype=np.float32
        )
        class_predictions = np.array(
            raw_outputs[self.class_predictions_out], dtype=np.float32
        )

        bounding_boxes, classes = self.get_bounding_boxes(
            np.int32(1), box_offsets, class_predictions,
            self.prior_boxes,
            np.array([.1, .1, .2, .2], dtype=np.float32), np.float32(0.01)
        )
        bounding_boxes, classes = self.nms(bounding_boxes, classes, 0.45)

        results = []
        for identifier, cur_bounding_boxes in zip(identifiers, bounding_boxes):
            x_mins, y_mins, x_maxs, y_maxs, labels, scores = [], [], [], [], \
                                                             [], []
            for x_min, x_max, y_min, y_max, confidence in cur_bounding_boxes:
                x_mins.append(x_min)
                y_mins.append(y_min)
                x_maxs.append(x_max)
                y_maxs.append(y_max)
                labels.append(1)
                scores.append(confidence)
            results.append(
                DetectionPrediction(
                    identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs
                )
            )
        return results

