
.. _l-onnx-doc-MaxUnpool:

=========
MaxUnpool
=========

.. contents::
    :local:


.. _l-onnx-op-maxunpool-11:

MaxUnpool - 11
==============

**Version**

* **name**: `MaxUnpool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxUnpool>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corrsponding
 pooling op that the unpooling op is trying to invert.

**Attributes**

* **kernel_shape** (required):
  The size of the kernel along each axis.
* **pads**:
  Padding for the beginning and ending along each spatial axis, it can
  take any value greater than or equal to 0. The value represent the
  number of pixels added to the beginning and end part of the
  corresponding axis. `pads` format should be as follow [x1_begin,
  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
  added at the beginning of axis `i` and xi_end, the number of pixels
  added at the end of axis `i`. This attribute cannot be used
  simultaneously with auto_pad attribute. If not present, the padding
  defaults to 0 along start and end of each spatial axis.
* **strides**:
  Stride along each spatial axis. If not present, the stride defaults
  to 1 along each spatial axis.

**Inputs**

Between 2 and 3 inputs.

* **X** (heterogeneous) - **T1**:
  Input data tensor that has to be unpooled. This tensor is typically
  the first output of the MaxPool op.Dimensions for image case are (N
  x C x H x W), where N is the batch size, C is the number of
  channels, and H and W are the height and the width of the data. For
  non-image case, the dimensions are in the form of (N x C x D1 x D2
  ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
* **I** (heterogeneous) - **T2**:
  Input data tensor containing the indices corresponding to elements
  in the first input tensor X.This tensor is typically the second
  output of the MaxPool op.Dimensions must be the same as input tensor
  X. The indices are linear, i.e. computed considering the tensor as
  flattened 1-D tensor, assuming row-major storage. Also, the linear
  indices should not consider padding. So the values in indices are in
  the range [0, N x C x D1 x ... x Dn).
* **output_shape** (optional, heterogeneous) - **T2**:
  The shape of the output can be explicitly set which will cause pads
  values to be auto generated. If 'output_shape' is specified, 'pads'
  values are ignored.

**Outputs**

* **output** (heterogeneous) - **T1**:
  Output data tensor that contains the result of the unpooling.

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **T2** in (
  tensor(int64)
  ):
  Constrain index tensor to int64

**Examples**

**without_output_shape**

::

    node = onnx.helper.make_node(
        'MaxUnpool',
        inputs=['xT', 'xI'],
        outputs=['y'],
        kernel_shape=[2, 2],
        strides=[2, 2]
    )
    xT = np.array([[[[1, 2],
                     [3, 4]]]], dtype=np.float32)
    xI = np.array([[[[5, 7],
                     [13, 15]]]], dtype=np.int64)
    y = np.array([[[[0, 0, 0, 0],
                    [0, 1, 0, 2],
                    [0, 0, 0, 0],
                    [0, 3, 0, 4]]]], dtype=np.float32)
    expect(node, inputs=[xT, xI], outputs=[y], name='test_maxunpool_export_without_output_shape')

**with_output_shape**

::

    node = onnx.helper.make_node(
        'MaxUnpool',
        inputs=['xT', 'xI', 'output_shape'],
        outputs=['y'],
        kernel_shape=[2, 2],
        strides=[2, 2]
    )
    xT = np.array([[[[5, 6],
                     [7, 8]]]], dtype=np.float32)
    xI = np.array([[[[5, 7],
                     [13, 15]]]], dtype=np.int64)
    output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
    y = np.array([[[[0, 0, 0, 0, 0],
                    [0, 5, 0, 6, 0],
                    [0, 0, 0, 0, 0],
                    [0, 7, 0, 8, 0],
                    [0, 0, 0, 0, 0]]]], dtype=np.float32)
    expect(node, inputs=[xT, xI, output_shape], outputs=[y], name='test_maxunpool_export_with_output_shape')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxUnpool essentially computes the partial inverse of the MaxPool op.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxUnpool essentially computes the partial inverse of the MaxPool op.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> The input information to this op is typically the output information from a MaxPool op. The first</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> The input information to this op is typically the output information from a MaxPool op. The first</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> The third (optional) input is a tensor that specifies the output size of the unpooling operation.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> The third (optional) input is a tensor that specifies the output size of the unpooling operation.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> the result of an unpooling operation should give back the original input to the unpooling op.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> the result of an unpooling operation should give back the original input to the unpooling op.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> The third input argument, output_size, is meant to disambiguate the op and produce output tensor of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> The third input argument, output_size, is meant to disambiguate the op and produce output tensor of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> known/predictable size.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> known/predictable size.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> which define the exact unpooling op. The attributes typically have the same values as the corrsponding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> which define the exact unpooling op. The attributes typically have the same values as the corrsponding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> pooling op that the unpooling op is trying to invert.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> pooling op that the unpooling op is trying to invert.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td></tr>
    <tr style="1px solid black;"><td><code>34</code></td><td><code>34</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Stride along each spatial axis.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Stride along each spatial axis.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  to 1 along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor that has to be unpooled. This tensor is typically</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor that has to be unpooled. This tensor is typically</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the first output of the MaxPool op.Dimensions for image case are (N</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the first output of the MaxPool op.Dimensions for image case are (N</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x C x H x W), where N is the batch size, C is the number of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x C x H x W), where N is the batch size, C is the number of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  channels, and H and W are the height and the width of the data. For</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  channels, and H and W are the height and the width of the data. For</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  non-image case, the dimensions are in the form of (N x C x D1 x D2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  non-image case, the dimensions are in the form of (N x C x D1 x D2</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ... Dn), where N is the batch size. Optionally, if dimension</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ... Dn), where N is the batch size. Optionally, if dimension</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** (heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** (heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor containing the indices corresponding to elements</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor containing the indices corresponding to elements</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  in the first input tensor X.This tensor is typically the second</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  in the first input tensor X.This tensor is typically the second</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output of the MaxPool op.Dimensions must be the same as input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output of the MaxPool op.Dimensions must be the same as input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  X. The indices are linear, i.e. computed considering the tensor as</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  X. The indices are linear, i.e. computed considering the tensor as</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  flattened 1-D tensor, assuming row-major storage. Also, the linear</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  flattened 1-D tensor, assuming row-major storage. Also, the linear</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices should not consider padding. So the values in indices are in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices should not consider padding. So the values in indices are in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the range [0, N x C x D1 x ... x Dn).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the range [0, N x C x D1 x ... x Dn).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_shape** (optional, heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_shape** (optional, heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The shape of the output can be explicitly set which will cause pads</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The shape of the output can be explicitly set which will cause pads</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values to be auto generated. If 'output_shape' is specified, 'pads'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values to be auto generated. If 'output_shape' is specified, 'pads'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are ignored.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are ignored.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor that contains the result of the unpooling.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor that contains the result of the unpooling.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td></tr>
    </table>

.. _l-onnx-op-maxunpool-9:

MaxUnpool - 9
=============

**Version**

* **name**: `MaxUnpool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxUnpool>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corrsponding
 pooling op that the unpooling op is trying to invert.

**Attributes**

* **kernel_shape** (required):
  The size of the kernel along each axis.
* **pads**:
  Padding for the beginning and ending along each spatial axis, it can
  take any value greater than or equal to 0. The value represent the
  number of pixels added to the beginning and end part of the
  corresponding axis. `pads` format should be as follow [x1_begin,
  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
  added at the beginning of axis `i` and xi_end, the number of pixels
  added at the end of axis `i`. This attribute cannot be used
  simultaneously with auto_pad attribute. If not present, the padding
  defaults to 0 along start and end of each spatial axis.
* **strides**:
  Stride along each spatial axis.

**Inputs**

Between 2 and 3 inputs.

* **X** (heterogeneous) - **T1**:
  Input data tensor that has to be unpooled. This tensor is typically
  the first output of the MaxPool op.Dimensions for image case are (N
  x C x H x W), where N is the batch size, C is the number of
  channels, and H and W are the height and the width of the data. For
  non-image case, the dimensions are in the form of (N x C x D1 x D2
  ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
* **I** (heterogeneous) - **T2**:
  Input data tensor containing the indices corresponding to elements
  in the first input tensor X.This tensor is typically the second
  output of the MaxPool op.Dimensions must be the same as input tensor
  X. The indices are linear, i.e. computed considering the tensor as
  flattened 1-D tensor, assuming row-major storage. Also, the linear
  indices should not consider padding. So the values in indices are in
  the range [0, N x C x D1 x ... x Dn).
* **output_shape** (optional, heterogeneous) - **T2**:
  The shape of the output can be explicitly set which will cause pads
  values to be auto generated. If 'output_shape' is specified, 'pads'
  values are ignored.

**Outputs**

* **output** (heterogeneous) - **T1**:
  Output data tensor that contains the result of the unpooling.

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **T2** in (
  tensor(int64)
  ):
  Constrain index tensor to int64
