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
import torch

from .metric import PerImageEvaluationMetric
from ..representation import SalientRegionAnnotation, SalientRegionPrediction


class SalienceMapMAE(PerImageEvaluationMetric):
    __provider__ = 'salience_mae'
    annotation_types = (SalientRegionAnnotation, )
    prediction_types = (SalientRegionPrediction, )

    def configure(self):
        self.errors = []
        self.meta.update({
            'names': ['mean', 'std'],
            'scale': 1,
            'postfix': ' ',
            'calculate_mean': False,
            'target': 'higher-worse',
            'data_format': "{:.4f}"
        })
        self.cnt = 0

    def update(self, annotation, prediction):
        self.cnt += 1
        # The original code is complicated and looks incorrect. We change it
        # to calculate the correct metric. Note that the maps are both 0-255
        # now, so we need to divide both maps by 255, or simply divide the
        # calculated MAE
        mae = np.mean(
            np.abs(
                annotation.mask.astype(np.float32) -
                prediction.mask.astype(np.float32)
            )
        ) / 255.0
        self.errors.append(mae)
        return mae

    def evaluate(self, annotations, predictions):
        return np.mean(self.errors), np.std(self.errors)

    def reset(self):
        del self.errors
        self.errors = []
        self.cnt = 0


class SalienceMapFMeasure(PerImageEvaluationMetric):
    __provider__ = 'salience_f-measure'
    annotation_types = (SalientRegionAnnotation, )
    prediction_types = (SalientRegionPrediction, )

    def configure(self):
        self.recalls, self.precisions, self.fmeasure = [], [], []
        self.meta.update({
            'names': ['f-measure', 'recall', 'precision'],
            'calculate_mean': False,
        })

    def update(self, annotation, prediction):
        # Both maps are now 0-255, so we divide both maps by 255 before using
        annotation_mask = annotation.mask.astype(np.float32) / 255.0
        prediction_mask = prediction.mask.astype(np.float32) / 255.0

        sum_label = 2 * np.mean(prediction_mask)
        if sum_label > 1:
            sum_label = 1

        label3 = np.zeros_like(annotation_mask)
        label3[prediction_mask >= sum_label] = 1

        num_recall = np.sum(label3 == 1)
        label_and = np.logical_and(label3, annotation_mask)
        num_and = np.sum(label_and == 1)
        num_obj = np.sum(annotation_mask)

        if num_and == 0:
            self.recalls.append(0)
            self.precisions.append(0)
            self.fmeasure.append(0)
            return 0
        precision = num_and / num_recall if num_recall != 0 else 0
        recall = num_and / num_obj if num_obj != 0 else 0
        fmeasure = (1.3 * precision * recall) / (0.3 * precision + recall) if precision + recall != 0 else 0
        self.recalls.append(recall)
        self.precisions.append(precision)
        self.fmeasure.append(fmeasure)
        return fmeasure

    def evaluate(self, annotations, predictions):
        return np.mean(self.fmeasure), np.mean(self.recalls), np.mean(self.precisions)

    def reset(self):
        self.recalls, self.precisions, self.fmeasure = [], [], []


class SalienceMapMaxFMeasure(PerImageEvaluationMetric):
    __provider__ = 'salience_max-f-measure'
    annotation_types = (SalientRegionAnnotation, )
    prediction_types = (SalientRegionPrediction, )

    def configure(self):
        self.recalls, self.precisions, self.fmeasure = [], [], []
        self.meta.update({
            'names': ['f-measure', 'recall', 'precision'],
            'calculate_mean': False,
        })

    def update(self, annotation, prediction):
        # Both maps are now 0-255, so we divide both maps by 255 before using
        annotation_mask = annotation.mask.astype(np.float32) / 255.0
        prediction_mask = prediction.mask.astype(np.float32) / 255.0

        # We use PyTorch CUDA to speedup evaluation, even though it is not
        # necessary
        sum_labels = torch.reshape(torch.arange(0, 256) / 255.0, (256, 1, 1)).cuda()
        prediction_mask = torch.tensor(prediction_mask).cuda()
        label3 = torch.tile(prediction_mask, (256, 1, 1)) >= sum_labels

        annotation_mask = torch.tensor(annotation_mask).cuda()
        mask = torch.unsqueeze(annotation_mask, 0)
        num_recalls = torch.sum(label3, dim=(1, 2))
        label_ands = torch.logical_and(label3, mask)
        num_ands = torch.sum(label_ands, dim=(1, 2))
        num_objs = torch.sum(annotation_mask)

        precisions = torch.divide(num_ands, num_recalls)
        precisions[precisions != precisions] = 0
        recalls = torch.div(num_ands, num_objs)
        recalls[recalls != recalls] = 0
        fmeasures = torch.div(
            1.3 * precisions * recalls, 0.3 * precisions + recalls
        )
        fmeasures[fmeasures != fmeasures] = 0

        precisions = precisions.cpu().detach()
        recalls = recalls.cpu().detach()
        fmeasures = fmeasures.cpu().detach()

        index = np.argmax(fmeasures)
        recall = recalls[index]
        precision = precisions[index]
        fmeasure = fmeasures[index]
        self.recalls.append(recall)
        self.precisions.append(precision)
        self.fmeasure.append(fmeasure)
        return fmeasure

    def evaluate(self, annotations, predictions):
        return np.mean(self.fmeasure), np.mean(self.recalls), np.mean(self.precisions)

    def reset(self):
        self.recalls, self.precisions, self.fmeasure = [], [], []


class SalienceEMeasure(PerImageEvaluationMetric):
    __provider__ = 'salience_e-measure'
    annotation_types = (SalientRegionAnnotation, )
    prediction_types = (SalientRegionPrediction, )

    def configure(self):
        self.scores = []

    def update(self, annotation, prediction):
        # Both maps are now 0-255, so we divide both maps by 255 before using
        annotation_mask = annotation.mask.astype(np.float32) / 255.0
        prediction_mask = prediction.mask.astype(np.float32) / 255.0

        if np.sum(annotation_mask) == 0:
            enhance_matrix = 1 - prediction_mask
        elif np.sum(~annotation_mask.astype(np.bool)) == 0:
            enhance_matrix = prediction_mask
        else:
            align_matrix = self.alignment_term(prediction_mask, annotation_mask)
            enhance_matrix = ((align_matrix + 1)**2) / 4
        h, w = annotation_mask.shape[:2]
        score = np.sum(enhance_matrix)/(w * h + np.finfo(float).eps)
        self.scores.append(score)
        return score

    @staticmethod
    def alignment_term(pred_mask, gt_mask):
        mu_fm = np.mean(pred_mask)
        mu_gt = np.mean(gt_mask)
        align_fm = pred_mask - mu_fm
        align_gt = gt_mask - mu_gt

        align_matrix = 2. * (align_gt * align_fm) / (align_gt * align_gt + align_fm * align_fm + np.finfo(float).eps)
        return align_matrix

    def evaluate(self, annotations, predictions):
        return np.mean(self.scores)

    def reset(self):
        self.scores = []


class SalienceSMeasure(PerImageEvaluationMetric):
    __provider__ = 'salience_s-measure'
    annotation_types = (SalientRegionAnnotation, )
    prediction_types = (SalientRegionPrediction, )

    def configure(self):
        self.scores = []

    def update(self, annotation, prediction):
        # Both maps are now 0-255, so we divide both maps by 255 before using
        annotation_mask = annotation.mask.astype(np.float32) / 255.0
        prediction_mask = prediction.mask.astype(np.float32) / 255.0

        y = np.mean(annotation_mask)
        x = np.mean(prediction_mask)
        if y == 0:
            self.scores.append(1 - x)
            return 1 - x
        if y == 1:
            self.scores.append(x)
            return x
        score = 0.5 * (
            self.s_object(prediction_mask, annotation_mask) +
            self.s_region(prediction_mask, annotation_mask)
        )
        self.scores.append(score)
        return score

    @staticmethod
    def s_object(pred_mask, gt_mask):
        def obj(pred, gt):
            x = np.mean(pred[np.where(gt)])
            x_sigma = np.std(pred(np.where(gt)))
            return 2 * x / (x ** 2 + 1 + x_sigma + np.finfo(float).eps)

        pred_fg = pred_mask
        pred_fg[np.where(~gt_mask)] = 0
        o_fg = obj(pred_fg, gt_mask)
        pred_bg = 1 - pred_mask
        pred_bg[np.where(gt_mask)] = 0
        o_bg = obj(pred_bg, ~gt_mask)
        union = np.mean(gt_mask)
        return union * o_fg + (1 - union) * o_bg

    def s_region(self, pred_mask, gt_mask):
        x, y = self.centroid(gt_mask)
        g1, g2, g3, g4, w1, w2, w3, w4 = self.get_regions(gt_mask, x, y, True)
        p1, p2, p3, p4 = self.get_regions(pred_mask, x, y)
        q1 = self.ssim(p1, g1)
        q2 = self.ssim(p2, g2)
        q3 = self.ssim(p3, g3)
        q4 = self.ssim(p4, g4)
        return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4

    @staticmethod
    def centroid(gt_mask):
        h, w = gt_mask.shape[:2]
        area = np.sum(gt_mask)
        if area == 0:
            return w // 2, h // 2
        x = np.ones((h, 1)) @ np.arange(w)
        y = np.arange(h).T @ np.ones((1, w))
        x = np.round(np.sum(x * gt_mask)) // area
        y = np.round(np.sum(y * gt_mask)) // area
        return x, y

    @staticmethod
    def get_regions(mask, x, y, get_width=False):
        height, width = mask.shape[:2]
        lt = mask[:y, :x]
        rt = mask[:y, (x + 1):]
        lb = mask[(y + 1):, :x]
        rb = mask[(y + 1):, (x + 1):]
        if not get_width:
            return lt, rt, lb, rb
        area = height * width
        w1 = x*y / area
        w2 = (width - x) * y / area
        w3 = (height - y) * x / area
        w4 = 1 - w1 - w2 - w3
        return lt, rt, lb, rb, w1, w2, w3, w4

    @staticmethod
    def ssim(pred, gt):
        h, w = pred.shape[:2]
        n = h * w
        x = np.mean(pred)
        y = np.mean(gt)
        sigma_x2 = np.sum((pred - x) ** 2) / (n - 1 + np.finfo(float).eps)
        sigma_y2 = np.sum((gt - y) ** 2) / (n - 1 + np.finfo(float).eps)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (n - 1 + np.finfo(float).eps)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

        if alpha != 0:
            return alpha / (beta + np.finfo(float).eps)
        if alpha == 0 and beta == 0:
            return 1.0
        return 0.0

    def evaluate(self, annotations, predictions):
        return np.mean(self.scores)

    def reset(self):
        self.scores = []
