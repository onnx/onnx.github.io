
.. _l-onnx-doc-MaxRoiPool:

==========
MaxRoiPool
==========

.. contents::
    :local:


.. _l-onnx-op-maxroipool-1:

MaxRoiPool - 1
==============

**Version**

* **name**: `MaxRoiPool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxRoiPool>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

ROI max pool consumes an input tensor X and region of interests (RoIs) to
apply max pooling across each RoI, to produce output 4-D tensor of shape
(num_rois, channels, pooled_shape[0], pooled_shape[1]).

**Attributes**

* **pooled_shape** (required):
  ROI pool output shape (height, width).
* **spatial_scale**:
  Multiplicative spatial scale factor to translate ROI coordinates
  from their input scale to the scale used when pooling. Default value is ``1.0``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
* **rois** (heterogeneous) - **T**:
  RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of
  shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].

**Outputs**

* **Y** (heterogeneous) - **T**:
  RoI pooled output 4-D tensor of shape (num_rois, channels,
  pooled_shape[0], pooled_shape[1]).

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**
