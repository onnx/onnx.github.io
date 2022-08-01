
.. _l-onnx-doccom.microsoft-QLinearAveragePool:

==================================
com.microsoft - QLinearAveragePool
==================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qlinearaveragepool-1:

QLinearAveragePool - 1 (com.microsoft)
======================================

**Version**

* **name**: `QLinearAveragePool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QLinearAveragePool>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

 QLinearAveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```

The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).

Input and output scales and zero points are used to convert the output to a new quantization range.
Output = Dequantize(Input) -> AveragePool on fp32 data -> Quantize(output)

**Attributes**

* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that the output
  spatial size match the input.In case of odd number add the extra
  padding at the end for SAME_UPPER and at the beginning for
  SAME_LOWER. VALID mean no padding. Default value is ``?``.
* **ceil_mode**:
  Whether to use ceil or floor (default) to compute the output shape. Default value is ``?``.
* **channels_last**:
  Works on NHWC layout or not? Default not. Default value is ``?``.
* **count_include_pad**:
  Whether include pad pixels when calculating values for the edges.
  Default is 0, doesn't count include pad. Default value is ``?``.
* **kernel_shape** (required):
  The size of the kernel along each axis. Default value is ``?``.
* **pads**:
  Padding for the beginning and ending along each spatial axis, it can
  take any value greater than or equal to 0. The value represent the
  number of pixels added to the beginning and end part of the
  corresponding axis. `pads` format should be as follow [x1_begin,
  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
  added at the beginning of axis `i` and xi_end, the number of pixels
  added at the end of axis `i`. This attribute cannot be used
  simultaneously with auto_pad attribute. If not present, the padding
  defaults to 0 along start and end of each spatial axis. Default value is ``?``.
* **strides**:
  Stride along each spatial axis. If not present, the stride defaults
  to 1 along each spatial axis. Default value is ``?``.

**Inputs**

Between 4 and 5 inputs.

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
  For non image case, the dimensions are in the form of (N x C x D1 x
  D2 ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
* **x_scale** (heterogeneous) - **tensor(float)**:
  Input scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **x_zero_point** (optional, heterogeneous) - **T**:
  Input zero point. Default value is 0 if it's not specified. It's a
  scalar, which means a per-tensor/layer quantization.
* **y_scale** (heterogeneous) - **tensor(float)**:
  Output scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **y_zero_point** (optional, heterogeneous) - **T**:
  Output zero point. Default value is 0 if it's not specified. It's a
  scalar, which means a per-tensor/layer quantization.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output data tensor from average or max pooling across the input
  tensor. Dimensions will vary based on various kernel, stride, and
  pad sizes. Floor value of the dimension is used

**Examples**
