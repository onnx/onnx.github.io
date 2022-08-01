
.. _l-onnx-doc-TopK:

====
TopK
====

.. contents::
    :local:


.. _l-onnx-op-topk-11:

TopK - 11
=========

**Version**

* **name**: `TopK (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).

If "largest" is 1 (the default value) then the k largest elements are returned.
If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
 a tiebreaker. That is, the element with the lower index will appear first.

**Attributes**

* **axis**:
  Dimension on which to do the sort. Negative value means counting
  dimensions from the back. Accepted range is [-r, r-1] where r =
  rank(input). Default value is ``-1``.
* **largest**:
  Whether to return the top-K largest or smallest elements. Default value is ``1``.
* **sorted**:
  Whether to return the elements in sorted order. Default value is ``1``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Tensor of shape [a_1, a_2, ..., a_n, r]
* **K** (heterogeneous) - **tensor(int64)**:
  A 1-D tensor containing a single positive value corresponding to the
  number of top elements to retrieve

**Outputs**

* **Values** (heterogeneous) - **T**:
  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
  containing top K values from the input tensor
* **Indices** (heterogeneous) - **I**:
  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
  containing the corresponding input tensor indices for the top K
  values.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input and output types to numeric tensors.
* **I** in (
  tensor(int64)
  ):
  Constrain index tensor to int64

**Examples**

**top_k**

::

    axis = 1
    largest = 1

    k = 3
    node = onnx.helper.make_node(
        'TopK',
        inputs=['x', 'k'],
        outputs=['values', 'indices'],
        axis=axis
    )
    X = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ], dtype=np.float32)
    K = np.array([k], dtype=np.int64)
    values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)

    #print(values_ref)
    #[[ 3.  2.  1.]
    # [ 7.  6.  5.]
    # [11. 10.  9.]]
    #print(indices_ref)
    #[[3 2 1]
    # [3 2 1]
    # [3 2 1]]

    expect(node, inputs=[X, K], outputs=[values_ref, indices_ref],
           name='test_top_k')

**top_k_smallest**

::

    axis = 1
    largest = 0
    sorted = 1
    k = 3

    node = onnx.helper.make_node(
        'TopK',
        inputs=['x', 'k'],
        outputs=['values', 'indices'],
        axis=axis,
        largest=largest,
        sorted=sorted
    )

    X = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [11, 10, 9, 8],
    ], dtype=np.float32)
    K = np.array([k], dtype=np.int64)
    values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)

    #print(values_ref)
    #[[ 0.  1.  2.]
    # [ 4.  5.  6.]
    # [ 8.  9. 10.]]
    #print(indices_ref)
    #[[0 1 2]
    # [0 1 2]
    # [3 2 1]]

    expect(node, inputs=[X, K], outputs=[values_ref, indices_ref],
           name='test_top_k_smallest')

**top_k_negative_axis**

::

    axis = -1
    largest = 1

    k = 3
    node = onnx.helper.make_node(
        'TopK',
        inputs=['x', 'k'],
        outputs=['values', 'indices'],
        axis=axis
    )
    X = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ], dtype=np.float32)
    K = np.array([k], dtype=np.int64)
    values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)

    # print(values_ref)
    #[[ 3.  2.  1.]
    # [ 7.  6.  5.]
    # [11. 10.  9.]]
    # print(indices_ref)
    #[[3 2 1]
    # [3 2 1]
    # [3 2 1]]

    expect(node, inputs=[X, K], outputs=[values_ref, indices_ref],
           name='test_top_k_negative_axis')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td><code>0</code></td><td><code>0</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Retrieve the top-K elements along a specified axis. Given an input tensor of</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Retrieve the top-K <span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span>l<span style="color:#196F3D;">l</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span>ments along a specified axis. Given an input tensor of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    which contains the values of the top k elements along the specified axis</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    which contains the values of the top k elements along the specified axis</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   contains the indices of the top k elements (original indices from the input</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   contains the indices of the top k elements (original indices from the input</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   tensor).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   tensor).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">If "largest" is 1 (the default value) then the k largest elements are returned.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">If "sorted" is 1 (the default value) then the resulting k elements will be sorted.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>8</code></td><td><code>12</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Given two equivalent values, this operator uses the indices along the axis <span style="color:#BA4A00;"> </span>as</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Given two equivalent values, this operator uses the indices along the axis as</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> a tiebreaker. That is, the element with the lower index will appear first.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> a tiebreaker. That is, the element with the lower index will appear first.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td></tr>
    <tr style="1px solid black;"><td><code>14</code></td><td><code>18</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Dimension on which to do the sort. <span style="color:#BA4A00;">D</span>e<span style="color:#BA4A00;">f</span>a<span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span>t value <span style="color:#BA4A00;">i</span>s <span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Dimension on which to do the sort. <span style="color:#196F3D;">N</span>e<span style="color:#196F3D;">g</span>at<span style="color:#196F3D;">i</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">e</span> value <span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span>s <span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">19</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  dimensions from the back. Accepted range is [-r, r-1] where r =</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">20</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  rank(input). Default value is -1.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **largest**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">22</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Whether to return the top-K largest or smallest elements. Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **sorted**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">24</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Whether to return the elements in sorted order. Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_n, r]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_n, r]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **K** (heterogeneous) - **tensor(int64)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **K** (heterogeneous) - **tensor(int64)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A 1-D tensor containing a single positive value corresponding to the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A 1-D tensor containing a single positive value corresponding to the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of top elements to retrieve</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of top elements to retrieve</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Values** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Values** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  containing top K values from the input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  containing top K values from the input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (heterogeneous) - **I**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (heterogeneous) - **I**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  containing the corresponding input tensor indices for the top K</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  containing the corresponding input tensor indices for the top K</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td><code>39</code></td><td><code>49</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  tensor(float16)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  tensor(float16)<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">50</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">51</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">52</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">53</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">54</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">55</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">56</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">57</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td><code>41</code></td><td><code>59</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Constrain input and output types to <span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span> tensors.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Constrain input and output types to <span style="color:#196F3D;">n</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span> tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td></tr>
    </table>

.. _l-onnx-op-topk-10:

TopK - 10
=========

**Version**

* **name**: `TopK (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).

Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.

**Attributes**

* **axis**:
  Dimension on which to do the sort. Default value is ``-1``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Tensor of shape [a_1, a_2, ..., a_n, r]
* **K** (heterogeneous) - **tensor(int64)**:
  A 1-D tensor containing a single positive value corresponding to the
  number of top elements to retrieve

**Outputs**

* **Values** (heterogeneous) - **T**:
  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
  containing top K values from the input tensor
* **Indices** (heterogeneous) - **I**:
  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
  containing the corresponding input tensor indices for the top K
  values.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **I** in (
  tensor(int64)
  ):
  Constrain index tensor to int64

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Retrieve the top-K elements along a specified axis. Given an input tensor of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Retrieve the top-K elements along a specified axis. Given an input tensor of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    which contains the values of the top k elements along the specified axis</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    which contains the values of the top k elements along the specified axis</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   contains the indices of the top k elements (original indices from the input</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   contains the indices of the top k elements (original indices from the input</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   tensor).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   tensor).</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">7</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Given two equivalent values, this operator uses the indices along the axis  as</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Given two equivalent values, this operator uses the indices along the axis  as</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> a tiebreaker. That is, the element with the lower index will appear first.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"> a tiebreaker. That is, the element with the lower index will appear first.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Dimension on which to do the sort. Default value is -1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Dimension on which to do the sort. Default value is -1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">14</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **k** (required):</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">15</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  Number of top elements to retrieve</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_n, r]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_n, r]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">20</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **K** (heterogeneous) - **tensor(int64)**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  A 1-D tensor containing a single positive value corresponding to the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">22</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  number of top elements to retrieve</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Values** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Values** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  containing top K values from the input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  containing top K values from the input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (heterogeneous) - **I**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (heterogeneous) - **I**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  containing the corresponding input tensor indices for the top K</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  containing the corresponding input tensor indices for the top K</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td></tr>
    </table>

.. _l-onnx-op-topk-1:

TopK - 1
========

**Version**

* **name**: `TopK (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.

**Attributes**

* **axis**:
  Dimension on which to do the sort. Default value is ``-1``.
* **k** (required):
  Number of top elements to retrieve

**Inputs**

* **X** (heterogeneous) - **T**:
  Tensor of shape [a_1, a_2, ..., a_n, r]

**Outputs**

* **Values** (heterogeneous) - **T**:
  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
  containing top K values from the input tensor
* **Indices** (heterogeneous) - **I**:
  Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
  containing the corresponding input tensor indices for the top K
  values.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **I** in (
  tensor(int64)
  ):
  Constrain index tensor to int64
