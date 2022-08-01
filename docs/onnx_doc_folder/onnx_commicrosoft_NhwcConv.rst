
.. _l-onnx-doccom.microsoft-NhwcConv:

========================
com.microsoft - NhwcConv
========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-nhwcconv-1:

NhwcConv - 1 (com.microsoft)
============================

**Version**

* **name**: `NhwcConv (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.NhwcConv>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **auto_pad**:
 Default value is ``?``.
* **dilations**:
  dilation value along each spatial axis of the filter. If not
  present, the dilation defaults is 1 along each spatial axis. Default value is ``?``.
* **group**:
  number of groups input channels and output channels are divided
  into. Default value is ``?``.
* **kernel_shape**:
  The shape of the convolution kernel. If not present, should be
  inferred from input W. Default value is ``?``.
* **pads**:
 Default value is ``?``.
* **strides**:
  Stride along each spatial axis. If not present, the stride defaults
  is 1 along each spatial axis. Default value is ``?``.

**Inputs**

Between 2 and 3 inputs.

* **X** (heterogeneous) - **T**:
  Input data tensor from previous layer; has size (N x C x H x W),
  where N is the batch size, C is the number of channels, and H and W
  are the height and width. Note that this is for the 2D image.
  Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if
  dimension denotation is in effect, the operation expects input data
  tensor to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
* **W** (heterogeneous) - **T**:
  The weight tensor that will be used in the convolutions; has size (M
  x C/group x kH x kW), where C is the number of channels, and kH and
  kW are the height and width of the kernel, and M is the number of
  feature maps. For more than 2 dimensions, the kernel shape will be
  (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the
  dimension of the kernel. Optionally, if dimension denotation is in
  effect, the operation expects the weight tensor to arrive with the
  dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL,
  FILTER_SPATIAL, FILTER_SPATIAL ...]. Assuming zero based indices for
  the shape array, X.shape[1] == (W.shape[1] * group) == C and
  W.shape[0] mod G == 0. Or in other words FILTER_IN_CHANNEL
  multiplied by the number of groups should be equal to DATA_CHANNEL
  and the number of feature maps M should be a multiple of the number
  of groups G.
* **B** (optional, heterogeneous) - **T**:
  Optional 1D bias to be added to the convolution, has size of M.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output data tensor that contains the result of the convolution. The
  output dimensions are functions of the kernel size, stride size, and
  pad lengths.

**Examples**
