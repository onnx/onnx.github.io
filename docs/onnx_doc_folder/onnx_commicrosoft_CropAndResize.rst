
.. _l-onnx-doccom.microsoft-CropAndResize:

=============================
com.microsoft - CropAndResize
=============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-cropandresize-1:

CropAndResize - 1 (com.microsoft)
=================================

**Version**

* **name**: `CropAndResize (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.CropAndResize>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Extracts crops from the input image tensor and resizes them using bilinear sampling or nearest neighbor sampling
(possibly with aspect ratio change) to a common output size specified by crop_height and crop_width.
Returns a tensor with crops from the input image at positions defined at the bounding box locations in boxes.
The cropped boxes are all resized (with bilinear or nearest neighbor interpolation) to
a fixed size = [crop_height, crop_width]. The result is a 4-D tensor [num_boxes, crop_height, crop_width, depth].
The resizing is corner aligned.

**Attributes**

* **extrapolation_value**:
  Value used for extrapolation, when applicable. Default is 0.0f. Default value is ``?``.
* **mode**:
  The pooling method. Two modes are supported: 'bilinear' and
  'nearest'. Default is 'bilinear'. Default value is ``?``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  Input data tensor from the previous operator; 4-D feature map of
  shape (N, C, H, W), where N is the batch size, C is the number of
  channels, and H and W are the height and the width of the data.
* **rois** (heterogeneous) - **T1**:
  RoIs (Regions of Interest) to pool over; rois is 2-D input of shape
  (num_rois, 4) given as [[y1, x1, y2, x2], ...]. The RoIs'
  coordinates are normalized in the coordinate system of the input
  image. Each coordinate set has a 1:1 correspondence with the
  'batch_indices' input.
* **batch_indices** (heterogeneous) - **T2**:
  1-D tensor of shape (num_rois,) with each element denoting the index
  of the corresponding image in the batch.
* **crop_size** (heterogeneous) - **T2**:
  1-D tensor of 2 elements: [crop_height, crop_width]. All cropped
  image patches are resized to this size. Both crop_height and
  crop_width need to be positive.

**Outputs**

* **Y** (heterogeneous) - **T1**:
  RoI pooled output, 4-D tensor of shape (num_rois, C, crop_height,
  crop_width). The r-th batch element Y[r-1] is a pooled feature map
  corresponding to the r-th RoI X[r-1].

**Examples**
