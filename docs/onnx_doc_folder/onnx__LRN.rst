
.. _l-onnx-doc-LRN:

===
LRN
===

.. contents::
    :local:


.. _l-onnx-op-lrn-13:

LRN - 13
========

**Version**

* **name**: `LRN (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.

square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta

**Attributes**

* **alpha**:
  Scaling parameter. Default value is ``9.999999747378752e-05``.
* **beta**:
  The exponent. Default value is ``0.75``.
* **bias**:
 Default value is ``1.0``.
* **size** (required):
  The number of channels to sum over

**Inputs**

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
  For non image case, the dimensions are in the form of (N x C x D1 x
  D2 ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor, which has the shape and type as input tensor

**Type Constraints**

* **T** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output  types to float tensors.

**Examples**

**default**

::

    alpha = 0.0001
    beta = 0.75
    bias = 1.0
    nsize = 3
    node = onnx.helper.make_node(
        'LRN',
        inputs=['x'],
        outputs=['y'],
        size=3
    )
    x = np.random.randn(5, 5, 5, 5).astype(np.float32)
    square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
    for n, c, h, w in np.ndindex(x.shape):
        square_sum[n, c, h, w] = sum(x[n,
                                       max(0, c - int(math.floor((nsize - 1) / 2))):min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                       h,
                                       w] ** 2)
    y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
    expect(node, inputs=[x], outputs=[y],
           name='test_lrn_default')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">It normalizes over local input regions.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">It normalizes over local input regions.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">of shape (N x C x D1 x D2, ..., Dk), its region is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">of shape (N x C x D1 x D2, ..., Dk), its region is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scaling parameter. Default value is 9.999999747378752e-05.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scaling parameter. Default value is 9.999999747378752e-05.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The exponent. Default value is 0.75.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The exponent. Default value is 0.75.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **bias**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **bias**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **size** (required):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **size** (required):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The number of channels to sum over</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The number of channels to sum over</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor, which has the shape and type as input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor, which has the shape and type as input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">42</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output  types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output  types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-lrn-1:

LRN - 1
=======

**Version**

* **name**: `LRN (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.

square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta

**Attributes**

* **alpha**:
  Scaling parameter. Default value is ``9.999999747378752e-05``.
* **beta**:
  The exponent. Default value is ``0.75``.
* **bias**:
 Default value is ``1.0``.
* **size** (required):
  The number of channels to sum over

**Inputs**

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
  For non image case, the dimensions are in the form of (N x C x D1 x
  D2 ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor, which has the shape and type as input tensor

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output  types to float tensors.
