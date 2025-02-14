# Image Processing C++ Demo

This demo processes the image according to the selected type of processing. The demo can work with the next types:

* `super_resolution`
* `deblurring`
* `jpeg_restoration`
* `style_transfer`

## Examples

All images on result frame will be marked one of these flags:

* 'O' - original image.
* 'R' - result image.
* 'D' - difference image (|result - original|).

1. Exmaple for deblurring type (left - source image, right - image after deblurring):

![](./assets/image_processing_deblurred_image.png)

2. Example for super_resolution type:

Low resolution:

![](./assets/image_processing_street_640x360.png)

Bicubic interpolation:

![](./assets/street_resized.png)

Super resolution:

![](./assets/street_resolution.png)

3. Example for jpeg_restoration type:

![](./assets/parrots_restoration.png)

For this type of image processing user can use flag `-jc`. It allows to perform compression before the inference (usefull when user want to test model on high quality jpeg images).

4. Example for style_transfer:

![](./assets/style_transfer.jpg)

## How It Works

Before running the demo, user must choose type of processing and model for this processing.\
For `super_resolution` user can choose the next models:

* [single-image-super-resolution-1032](../../../models/intel/single-image-super-resolution-1032/README.md) enhances the resolution of the input image by a factor of 4.
* [single-image-super-resolution-1033](../../../models/intel/single-image-super-resolution-1033/README.md) enhances the resolution of the input image by a factor of 3.
* [text-image-super-resolution-0001](../../../models/intel/text-image-super-resolution-0001/README.md) - a tiny model to 3x upscale scanned images with text.

For `deblurring` user can use [deblurgan-v2](../../../models/public/deblurgan-v2/README.md) - generative adversarial network for single image motion deblurring.

For `jpeg_restoration` user can use [fbcnn](../../../models/public/fbcnn/README.md) - flexible blind convolutional neural network for JPEG artifacts removal.

For `style_transfer` user can use [fast-neural-style-mosaic-onnx](../../../models/public/fast-neural-style-mosaic-onnx/README.md) - one of the style transfer models designed to mix the content of an image with the style of another image.

The demo runs inference and shows results for each image captured from an input. Depending on number of inference requests processing simultaneously (-nireq parameter) the pipeline might minimize the time required to process each single image (for nireq 1) or maximizes utilization of the device and overall processing performance.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html#general-conversion-parameters).

## Preparing to Run

For demo input image or video files, refer to the section **Media Files Available for Demos** in the [Open Model Zoo Demos Overview](../../README.md).
The list of models supported by the demo is in `<omz_dir>/demos/segmentation_demo/cpp/models.lst` file.
This file can be used as a parameter for [Model Downloader](../../../tools/model_tools/README.md) and Converter to download and, if necessary, convert models to OpenVINO IR format (\*.xml + \*.bin).

### Supported Models

* single-image-super-resolution-1032
* single-image-super-resolution-1033
* text-image-super-resolution-0001
* deblurgan-v2
* fbcnn
* fast-neural-style-mosaic-onnx

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running

Running the demo with the `-h` option yields a usage message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../../tools/model_tools/README.md). The list of models supported by the demo is in `<omz_dir>/demos/image_processing_demo/cpp/models.lst`.

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvino.ai/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

You can use the following command to enhance the resolution of the images captured by a camera using a pre-trained single-image-super-resolution-1033 network:

```sh
./image_processing_demo -i 0 -m single-image-super-resolution-1033.xml -at sr
```

### Modes

Demo application supports 3 modes:

1. to display the result image.
2. to display the original image with result together (left part is result, right part is original). Position of separator is specified be slider of trackbar.
3. to display the difference image with result together. By analogy with second mode.

User is able to change mode in run time using the next keys:

* **R** to display the result.
* **O** to display the original image with result.
* **V** to display the difference image with result.
* **Esc or Q** to quit

You can save processed results to a Motion JPEG AVI file or separate JPEG or PNG files using the `-o` option:

* To save processed results in an AVI file, specify the name of the output file with `avi` extension, for example: `-o output.avi`.
* To save processed results as images, specify the template name of the output image file with `jpg` or `png` extension, for example: `-o output_%03d.jpg`. The actual file names are constructed from the template at runtime by replacing regular expression `%03d` with the frame number, resulting in the following: `output_000.jpg`, `output_001.jpg`, and so on.
To avoid disk space overrun in case of continuous input stream, like camera, you can limit the amount of data stored in the output file(s) with the `limit` option. The default value is 1000. To change it, you can apply the `-limit N` option, where `N` is the number of frames to store.

>**NOTE**: Windows\* systems may not have the Motion JPEG codec installed by default. If this is the case, you can download OpenCV FFMPEG back end using the PowerShell script provided with the OpenVINO &trade; install package and located at `<INSTALL_DIR>/opencv/ffmpeg-download.ps1`. The script should be run with administrative privileges if OpenVINO &trade; is installed in a system protected folder (this is a typical case). Alternatively, you can save results as images.

## Demo Output

The demo uses OpenCV to display and write the resulting images. The demo reports:

* **FPS**: average rate of video frame processing (frames per second).
* **Latency**: average time required to process one frame (from reading the frame to displaying the results).
* Latency for each of the following pipeline stages:
  * **Decoding** — capturing input data.
  * **Preprocessing** — data preparation for inference.
  * **Inference** — infering input data (images) and getting a result.
  * **Postrocessing** — preparation inference result for output.
  * **Rendering** — generating output image.

You can use these metrics to measure application-level performance.

## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
