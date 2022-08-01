
.. _l-onnx-doc-RoiAlign:

========
RoiAlign
========

.. contents::
    :local:


.. _l-onnx-op-roialign-16:

RoiAlign - 16
=============

**Version**

* **name**: `RoiAlign (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign>`_
* **domain**: **main**
* **since_version**: **16**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 16**.

**Summary**

Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.

**Attributes**

* **coordinate_transformation_mode**:
  Allowed values are 'half_pixel' and 'output_half_pixel'. Use the
  value 'half_pixel' to pixel shift the input coordinates by -0.5 (the
  recommended behavior). Use the value 'output_half_pixel' to omit the
  pixel shift for the input (use this for a backward-compatible
  behavior). Default value is ``'half_pixel'``.
* **mode**:
  The pooling method. Two modes are supported: 'avg' and 'max'.
  Default is 'avg'. Default value is ``'avg'``.
* **output_height**:
  default 1; Pooled output Y's height. Default value is ``1``.
* **output_width**:
  default 1; Pooled output Y's width. Default value is ``1``.
* **sampling_ratio**:
  Number of sampling points in the interpolation grid used to compute
  the output value of each pooled output bin. If > 0, then exactly
  sampling_ratio x sampling_ratio grid points are used. If == 0, then
  an adaptive number of grid points are used (computed as
  ceil(roi_width / output_width), and likewise for height). Default is
  0. Default value is ``0``.
* **spatial_scale**:
  Multiplicative spatial scale factor to translate ROI coordinates
  from their input spatial scale to the scale used when pooling, i.e.,
  spatial scale of the input feature map X relative to the input
  image. E.g.; default is 1.0f. Default value is ``1.0``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  Input data tensor from the previous operator; 4-D feature map of
  shape (N, C, H, W), where N is the batch size, C is the number of
  channels, and H and W are the height and the width of the data.
* **rois** (heterogeneous) - **T1**:
  RoIs (Regions of Interest) to pool over; rois is 2-D input of shape
  (num_rois, 4) given as [[x1, y1, x2, y2], ...]. The RoIs'
  coordinates are in the coordinate system of the input image. Each
  coordinate set has a 1:1 correspondence with the 'batch_indices'
  input.
* **batch_indices** (heterogeneous) - **T2**:
  1-D tensor of shape (num_rois,) with each element denoting the index
  of the corresponding image in the batch.

**Outputs**

* **Y** (heterogeneous) - **T1**:
  RoI pooled output, 4-D tensor of shape (num_rois, C, output_height,
  output_width). The r-th batch element Y[r-1] is a pooled feature map
  corresponding to the r-th RoI X[r-1].

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain types to float tensors.
* **T2** in (
  tensor(int64)
  ):
  Constrain types to int tensors.

**Examples**

**roialign_aligned_false**

::

    node = onnx.helper.make_node(
        "RoiAlign",
        inputs=["X", "rois", "batch_indices"],
        outputs=["Y"],
        spatial_scale=1.0,
        output_height=5,
        output_width=5,
        sampling_ratio=2,
        coordinate_transformation_mode="output_half_pixel",
    )

    X, batch_indices, rois = get_roi_align_input_values()
    # (num_rois, C, output_height, output_width)
    Y = np.array(
        [
            [
                [
                    [0.4664, 0.4466, 0.3405, 0.5688, 0.6068],
                    [0.3714, 0.4296, 0.3835, 0.5562, 0.3510],
                    [0.2768, 0.4883, 0.5222, 0.5528, 0.4171],
                    [0.4713, 0.4844, 0.6904, 0.4920, 0.8774],
                    [0.6239, 0.7125, 0.6289, 0.3355, 0.3495],
                ]
            ],
            [
                [
                    [0.3022, 0.4305, 0.4696, 0.3978, 0.5423],
                    [0.3656, 0.7050, 0.5165, 0.3172, 0.7015],
                    [0.2912, 0.5059, 0.6476, 0.6235, 0.8299],
                    [0.5916, 0.7389, 0.7048, 0.8372, 0.8893],
                    [0.6227, 0.6153, 0.7097, 0.6154, 0.4585],
                ]
            ],
            [
                [
                    [0.2384, 0.3379, 0.3717, 0.6100, 0.7601],
                    [0.3767, 0.3785, 0.7147, 0.9243, 0.9727],
                    [0.5749, 0.5826, 0.5709, 0.7619, 0.8770],
                    [0.5355, 0.2566, 0.2141, 0.2796, 0.3600],
                    [0.4365, 0.3504, 0.2887, 0.3661, 0.2349],
                ]
            ],
        ],
        dtype=np.float32,
    )

    expect(node, inputs=[X, rois, batch_indices], outputs=[Y], name="test_roialign_aligned_false")

**roialign_aligned_true**

::

    node = onnx.helper.make_node(
        "RoiAlign",
        inputs=["X", "rois", "batch_indices"],
        outputs=["Y"],
        spatial_scale=1.0,
        output_height=5,
        output_width=5,
        sampling_ratio=2,
        coordinate_transformation_mode="half_pixel",
    )

    X, batch_indices, rois = get_roi_align_input_values()
    # (num_rois, C, output_height, output_width)
    Y = np.array(
        [
            [
                [
                    [0.5178, 0.3434, 0.3229, 0.4474, 0.6344],
                    [0.4031, 0.5366, 0.4428, 0.4861, 0.4023],
                    [0.2512, 0.4002, 0.5155, 0.6954, 0.3465],
                    [0.3350, 0.4601, 0.5881, 0.3439, 0.6849],
                    [0.4932, 0.7141, 0.8217, 0.4719, 0.4039],
                ]
            ],
            [
                [
                    [0.3070, 0.2187, 0.3337, 0.4880, 0.4870],
                    [0.1871, 0.4914, 0.5561, 0.4192, 0.3686],
                    [0.1433, 0.4608, 0.5971, 0.5310, 0.4982],
                    [0.2788, 0.4386, 0.6022, 0.7000, 0.7524],
                    [0.5774, 0.7024, 0.7251, 0.7338, 0.8163],
                ]
            ],
            [
                [
                    [0.2393, 0.4075, 0.3379, 0.2525, 0.4743],
                    [0.3671, 0.2702, 0.4105, 0.6419, 0.8308],
                    [0.5556, 0.4543, 0.5564, 0.7502, 0.9300],
                    [0.6626, 0.5617, 0.4813, 0.4954, 0.6663],
                    [0.6636, 0.3721, 0.2056, 0.1928, 0.2478],
                ]
            ],
        ],
        dtype=np.float32,
    )

    expect(node, inputs=[X, rois, batch_indices], outputs=[Y], name="test_roialign_aligned_true")

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Region of Interest (RoI) align operation described in the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Region of Interest (RoI) align operation described in the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RoiAlign consumes an input tensor X and region of interests (rois)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RoiAlign consumes an input tensor X and region of interests (rois)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">to apply pooling across each RoI; it produces a 4-D tensor of shape</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">to apply pooling across each RoI; it produces a 4-D tensor of shape</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(num_rois, C, output_height, output_width).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(num_rois, C, output_height, output_width).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RoiAlign is proposed to avoid the misalignment by removing</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RoiAlign is proposed to avoid the misalignment by removing</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">quantizations while converting from original image into feature</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">quantizations while converting from original image into feature</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">map and from feature map into RoI feature; in each ROI bin,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">map and from feature map into RoI feature; in each ROI bin,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the value of the sampled locations are computed directly</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the value of the sampled locations are computed directly</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">through bilinear interpolation.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">through bilinear interpolation.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">14</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **coordinate_transformation_mode**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">15</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Allowed values are 'half_pixel' and 'output_half_pixel'. Use the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">16</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  value 'half_pixel' to pixel shift the input coordinates by -0.5 (the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">17</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  recommended behavior). Use the value 'output_half_pixel' to omit the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">18</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  pixel shift for the input (use this for a backward-compatible</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">19</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  behavior). Default value is 'half_pixel'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mode**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mode**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The pooling method. Two modes are supported: 'avg' and 'max'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The pooling method. Two modes are supported: 'avg' and 'max'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Default is 'avg'. Default value is 'avg'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Default is 'avg'. Default value is 'avg'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_height**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_height**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  default 1; Pooled output Y's height. Default value is 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  default 1; Pooled output Y's height. Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_width**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_width**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  default 1; Pooled output Y's width. Default value is 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  default 1; Pooled output Y's width. Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sampling_ratio**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sampling_ratio**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of sampling points in the interpolation grid used to compute</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of sampling points in the interpolation grid used to compute</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the output value of each pooled output bin. If > 0, then exactly</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the output value of each pooled output bin. If > 0, then exactly</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  sampling_ratio x sampling_ratio grid points are used. If == 0, then</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  sampling_ratio x sampling_ratio grid points are used. If == 0, then</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  an adaptive number of grid points are used (computed as</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  an adaptive number of grid points are used (computed as</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ceil(roi_width / output_width), and likewise for height). Default is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ceil(roi_width / output_width), and likewise for height). Default is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **spatial_scale**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **spatial_scale**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Multiplicative spatial scale factor to translate ROI coordinates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Multiplicative spatial scale factor to translate ROI coordinates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  from their input spatial scale to the scale used when pooling, i.e.,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  from their input spatial scale to the scale used when pooling, i.e.,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial scale of the input feature map X relative to the input</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial scale of the input feature map X relative to the input</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  image. E.g.; default is 1.0f. Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  image. E.g.; default is 1.0f. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; 4-D feature map of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; 4-D feature map of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape (N, C, H, W), where N is the batch size, C is the number of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape (N, C, H, W), where N is the batch size, C is the number of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  channels, and H and W are the height and the width of the data.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  channels, and H and W are the height and the width of the data.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **rois** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **rois** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  RoIs (Regions of Interest) to pool over; rois is 2-D input of shape</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  RoIs (Regions of Interest) to pool over; rois is 2-D input of shape</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (num_rois, 4) given as [[x1, y1, x2, y2], ...]. The RoIs'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (num_rois, 4) given as [[x1, y1, x2, y2], ...]. The RoIs'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinates are in the coordinate system of the input image. Each</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinates are in the coordinate system of the input image. Each</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate set has a 1:1 correspondence with the 'batch_indices'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate set has a 1:1 correspondence with the 'batch_indices'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  input.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  input.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **batch_indices** (heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **batch_indices** (heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of shape (num_rois,) with each element denoting the index</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of shape (num_rois,) with each element denoting the index</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of the corresponding image in the batch.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of the corresponding image in the batch.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  RoI pooled output, 4-D tensor of shape (num_rois, C, output_height,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  RoI pooled output, 4-D tensor of shape (num_rois, C, output_height,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output_width). The r-th batch element Y[r-1] is a pooled feature map</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output_width). The r-th batch element Y[r-1] is a pooled feature map</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding to the r-th RoI X[r-1].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding to the r-th RoI X[r-1].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain types to int tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain types to int tensors.</code></td></tr>
    </table>

.. _l-onnx-op-roialign-10:

RoiAlign - 10
=============

**Version**

* **name**: `RoiAlign (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.

**Attributes**

* **mode**:
  The pooling method. Two modes are supported: 'avg' and 'max'.
  Default is 'avg'. Default value is ``'avg'``.
* **output_height**:
  default 1; Pooled output Y's height. Default value is ``1``.
* **output_width**:
  default 1; Pooled output Y's width. Default value is ``1``.
* **sampling_ratio**:
  Number of sampling points in the interpolation grid used to compute
  the output value of each pooled output bin. If > 0, then exactly
  sampling_ratio x sampling_ratio grid points are used. If == 0, then
  an adaptive number of grid points are used (computed as
  ceil(roi_width / output_width), and likewise for height). Default is
  0. Default value is ``0``.
* **spatial_scale**:
  Multiplicative spatial scale factor to translate ROI coordinates
  from their input spatial scale to the scale used when pooling, i.e.,
  spatial scale of the input feature map X relative to the input
  image. E.g.; default is 1.0f. Default value is ``1.0``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  Input data tensor from the previous operator; 4-D feature map of
  shape (N, C, H, W), where N is the batch size, C is the number of
  channels, and H and W are the height and the width of the data.
* **rois** (heterogeneous) - **T1**:
  RoIs (Regions of Interest) to pool over; rois is 2-D input of shape
  (num_rois, 4) given as [[x1, y1, x2, y2], ...]. The RoIs'
  coordinates are in the coordinate system of the input image. Each
  coordinate set has a 1:1 correspondence with the 'batch_indices'
  input.
* **batch_indices** (heterogeneous) - **T2**:
  1-D tensor of shape (num_rois,) with each element denoting the index
  of the corresponding image in the batch.

**Outputs**

* **Y** (heterogeneous) - **T1**:
  RoI pooled output, 4-D tensor of shape (num_rois, C, output_height,
  output_width). The r-th batch element Y[r-1] is a pooled feature map
  corresponding to the r-th RoI X[r-1].

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain types to float tensors.
* **T2** in (
  tensor(int64)
  ):
  Constrain types to int tensors.
