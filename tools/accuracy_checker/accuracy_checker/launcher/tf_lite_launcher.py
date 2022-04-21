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
import os
import threading
from multiprocessing.dummy import Pool

import numpy as np
import tflite_runtime.interpreter

from edgetpu_pass.model.tflite_model import TFLiteModel
from edgetpu_pass.utils.misc import get_num_cpu
from .launcher import Launcher, ListInputsField
from ..config import PathField, StringField, NumberField


class TFLiteLauncher(Launcher):
    __provider__ = 'tf_lite'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'inputs': ListInputsField(optional=True),
            'model': PathField(is_directory=False, description="Path to model."),
            # Modified to support Edge TPU
            'device': StringField(choices=('CPU', 'TPU'), optional=True,
                                  description="Device: CPU or TPU."),
            "batch": NumberField(
                default=1,
                description="Batch size of input."
            )
        })
        return parameters

    def __init__(self, config_entry, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        # Change to use TFLiteModel directly
        use_edgetpu = self.config.get("device") == "TPU"

        model_args = {
            "use_edgetpu": use_edgetpu,
        }
        if use_edgetpu:
            self.pool_threads = 1
        else:
            model_args["num_threads"] = 1
            self.pool_threads = os.cpu_count()
        thread_local = threading.local()

        def initializer():
            thread_local.model = self.model = TFLiteModel(
                str(self.config["model"]),
                **model_args
            )

        def run(inputs):
            return thread_local.model(*inputs)

        self.pool = Pool(self.pool_threads, initializer=initializer)
        self.run = run

        self.tf_lite = tflite_runtime.interpreter
        self.default_layout = 'NHWC'
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)

        self.validate_config(config_entry, delayed_model_loading=self._delayed_model_loading)
        # We ignore _delayed_model_loading and load the model regardless,
        # otherwise error is raised when other methods are called
        self._input_details = self.model.get_interpreter().get_input_details()
        self._output_details = self.model.get_interpreter().get_output_details()
        # We change this to record the input shapes only, as the original
        # code records all input details, which is not what the property
        # "inputs" should return
        self._inputs = {
            input_layer['name']: input_layer["shape"]
            for input_layer in self._input_details
        }
        self.device = '/{}:0'.format(self.config.get('device', 'cpu').lower())

    def predict(self, inputs, metadata=None, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """

        results = []

        for dataset_input in inputs:
            list_inputs = []
            for detail in self._input_details:
                input_tensor_name = detail["name"]
                cur_input = dataset_input[input_tensor_name].astype(np.float32)
                batch_size = cur_input.shape[0]
                list_inputs.append(cur_input)
            pool_inputs = []
            for i in range(batch_size):
                pool_inputs.append([
                    cur_input[i:i + 1] for cur_input in list_inputs
                ])
            pool_outputs = self.pool.map(self.run, pool_inputs)
            list_results = []
            for i, _ in enumerate(self._output_details):
                list_results.append(np.concatenate([
                    cur_output[i] for cur_output in pool_outputs
                ], axis=0))

            res = dict()
            for tensor, detail in zip(list_results, self._output_details):
                res[detail["name"]] = tensor
            results.append(res)

            if metadata is not None:
                for meta_ in metadata:
                    meta_['input_shape'] = self.inputs_info_for_meta()

        return results

    @property
    def batch(self):
        batch_size = self.config["batch"] if "batch" in self.config else 1
        return batch_size

    @property
    def inputs(self):
        return self._inputs

    def release(self):
        del self.model

    def predict_async(self, *args, **kwargs):
        raise ValueError('TensorFlow Lite Launcher does not support async mode yet')

    @property
    def output_blob(self):
        return next(iter(self._output_details))['name']
