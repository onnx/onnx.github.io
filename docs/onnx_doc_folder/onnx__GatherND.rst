
.. _l-onnx-doc-GatherND:

========
GatherND
========

.. contents::
    :local:


.. _l-onnx-op-gathernd-13:

GatherND - 13
=============

**Version**

* **name**: `GatherND (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r-b` => error condition

2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

`Example 1`

  batch_dims = 0

  data    = [[0,1],[2,3]]   # data_shape = [2, 2]

  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]

  output  = [0,3]           # output_shape = [2]

`Example 2`

  batch_dims = 0

  data    = [[0,1],[2,3]]  # data_shape = [2, 2]

  indices = [[1],[0]]      # indices_shape = [2, 1]

  output  = [[2,3],[0,1]]  # output_shape = [2, 2]

`Example 3`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]

  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]

`Example 4`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]

  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]

`Example 5`

  batch_dims = 1

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[1],[0]]             # indices_shape = [2, 1]

  output  = [[2,3],[4,5]]             # output_shape = [2, 2]

**Attributes**

* **batch_dims**:
  The number of batch dimensions. The gather of indexing starts from
  dimension of data[batch_dims:] Default value is ``0``.

**Inputs**

* **data** (heterogeneous) - **T**:
  Tensor of rank r >= 1.
* **indices** (heterogeneous) - **tensor(int64)**:
  Tensor of rank q >= 1. All index values are expected to be within
  bounds [-s, s-1] along axis of size s. It is an error if any of the
  index values are out of bounds.

**Outputs**

* **output** (heterogeneous) - **T**:
  Tensor of rank q + r - indices_shape[-1] - 1.

**Type Constraints**

* **T** in (
  tensor(bfloat16),
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input and output types to any tensor type.

**Examples**

**int32**

::

    node = onnx.helper.make_node(
        'GatherND',
        inputs=['data', 'indices'],
        outputs=['output'],
    )

    data = np.array([[0, 1], [2, 3]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    output = gather_nd_impl(data, indices, 0)
    expected_output = np.array([0, 3], dtype=np.int32)
    assert (np.array_equal(output, expected_output))
    expect(node, inputs=[data, indices], outputs=[output],
           name='test_gathernd_example_int32')

**float32**

::

    node = onnx.helper.make_node(
        'GatherND',
        inputs=['data', 'indices'],
        outputs=['output'],
    )

    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
    output = gather_nd_impl(data, indices, 0)
    expected_output = np.array([[[2, 3]], [[4, 5]]], dtype=np.float32)
    assert (np.array_equal(output, expected_output))
    expect(node, inputs=[data, indices], outputs=[output],
           name='test_gathernd_example_float32')

**int32_batchdim_1**

::

    node = onnx.helper.make_node(
        'GatherND',
        inputs=['data', 'indices'],
        outputs=['output'],
        batch_dims=1,
    )

    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    indices = np.array([[1], [0]], dtype=np.int64)
    output = gather_nd_impl(data, indices, 1)
    expected_output = np.array([[2, 3], [4, 5]], dtype=np.int32)
    assert (np.array_equal(output, expected_output))
    expect(node, inputs=[data, indices], outputs=[output],
           name='test_gathernd_example_int32_batch_dim1')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Given data tensor of rank r >= 1, indices tensor of rank q >= 1, and batch_dims integer b, this operator gathers</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Given data tensor of rank r >= 1, indices tensor of rank q >= 1, and batch_dims integer b, this operator gathers</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">slices of data into an output tensor of rank q + r - indices_shape[-1] - 1 - b.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">slices of data into an output tensor of rank q + r - indices_shape[-1] - 1 - b.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">indices is an q-dimensional integer tensor, best thought of as a (q-1)-dimensional tensor of index-tuples into data,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">indices is an q-dimensional integer tensor, best thought of as a (q-1)-dimensional tensor of index-tuples into data,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where each element defines a slice of data</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where each element defines a slice of data</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">batch_dims (denoted as b) is an integer indicating the number of batch dimensions, i.e the leading b number of dimensions of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">batch_dims (denoted as b) is an integer indicating the number of batch dimensions, i.e the leading b number of dimensions of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data tensor and indices are representing the batches, and the gather starts from the b+1 dimension.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data tensor and indices are representing the batches, and the gather starts from the b+1 dimension.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Some salient points about the inputs' rank and shape:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Some salient points about the inputs' rank and shape:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks r and q</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks r and q</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) The first b dimensions of the shape of indices tensor and data tensor must be equal.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) The first b dimensions of the shape of indices tensor and data tensor must be equal.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3) b < min(q, r) is to be honored.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3) b < min(q, r) is to be honored.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4) The indices_shape[-1] should have a value between 1 (inclusive) and rank r-b (inclusive)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4) The indices_shape[-1] should have a value between 1 (inclusive) and rank r-b (inclusive)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5) All values in indices are expected to be within bounds [-s, s-1] along axis of size s (i.e.) -data_shape[i] <= indices[...,i] <= data_shape[i] - 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5) All values in indices are expected to be within bounds [-s, s-1] along axis of size s (i.e.) -data_shape[i] <= indices[...,i] <= data_shape[i] - 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   It is an error if any of the index values are out of bounds.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   It is an error if any of the index values are out of bounds.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output is computed as follows:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output is computed as follows:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output tensor is obtained by mapping each index-tuple in the indices tensor to the corresponding slice of the input data.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output tensor is obtained by mapping each index-tuple in the indices tensor to the corresponding slice of the input data.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) If indices_shape[-1] > r-b => error condition</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) If indices_shape[-1] > r-b => error condition</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) If indices_shape[-1] == r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensors</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) If indices_shape[-1] == r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensors</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   containing 1-D tensors of dimension r-b, where N is an integer equals to the product of 1 and all the elements in the batch dimensions</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   containing 1-D tensors of dimension r-b, where N is an integer equals to the product of 1 and all the elements in the batch dimensions</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   of the indices_shape. Let us think of each such r-b ranked tensor as indices_slice. Each *scalar value* corresponding to data[0:b-1,indices_slice]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   of the indices_shape. Let us think of each such r-b ranked tensor as indices_slice. Each *scalar value* corresponding to data[0:b-1,indices_slice]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor (Example 1 below)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor (Example 1 below)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3) If indices_shape[-1] < r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3) If indices_shape[-1] < r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   containing 1-D tensors of dimension < r-b. Let us think of each such tensors as indices_slice. Each *tensor slice* corresponding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   containing 1-D tensors of dimension < r-b. Let us think of each such tensors as indices_slice. Each *tensor slice* corresponding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   to data[0:b-1, indices_slice , :] is filled into the corresponding location of the (q-b-1)-dimensional tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   to data[0:b-1, indices_slice , :] is filled into the corresponding location of the (q-b-1)-dimensional tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   to form the output tensor (Examples 2, 3, 4 and 5 below)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   to form the output tensor (Examples 2, 3, 4 and 5 below)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator is the inverse of ScatterND.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator is the inverse of ScatterND.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[0,1],[2,3]]   # data_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[0,1],[2,3]]   # data_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [0,3]           # output_shape = [2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [0,3]           # output_shape = [2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[0,1],[2,3]]  # data_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[0,1],[2,3]]  # data_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[1],[0]]      # indices_shape = [2, 1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[1],[0]]      # indices_shape = [2, 1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[0,1]]  # output_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[0,1]]  # output_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 3</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 4</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 5</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_dims = 1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[1],[0]]             # indices_shape = [2, 1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[1],[0]]             # indices_shape = [2, 1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[4,5]]             # output_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[4,5]]             # output_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **batch_dims**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **batch_dims**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The number of batch dimensions. The gather of indexing starts from</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The number of batch dimensions. The gather of indexing starts from</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension of data[batch_dims:] Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension of data[batch_dims:] Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >= 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >= 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **indices** (heterogeneous) - **tensor(int64)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **indices** (heterogeneous) - **tensor(int64)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank q >= 1. All index values are expected to be within</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank q >= 1. All index values are expected to be within</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  bounds [-s, s-1] along axis of size s. It is an error if any of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  bounds [-s, s-1] along axis of size s. It is an error if any of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  index values are out of bounds.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  index values are out of bounds.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank q + r - indices_shape[-1] - 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank q + r - indices_shape[-1] - 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">113</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to any tensor type.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to any tensor type.</code></td></tr>
    </table>

.. _l-onnx-op-gathernd-12:

GatherND - 12
=============

**Version**

* **name**: `GatherND (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND>`_
* **domain**: **main**
* **since_version**: **12**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 12**.

**Summary**

Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r-b` => error condition

2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

`Example 1`

  batch_dims = 0

  data    = [[0,1],[2,3]]   # data_shape = [2, 2]

  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]

  output  = [0,3]           # output_shape = [2]

`Example 2`

  batch_dims = 0

  data    = [[0,1],[2,3]]  # data_shape = [2, 2]

  indices = [[1],[0]]      # indices_shape = [2, 1]

  output  = [[2,3],[0,1]]  # output_shape = [2, 2]

`Example 3`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]

  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]

`Example 4`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]

  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]

`Example 5`

  batch_dims = 1

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[1],[0]]             # indices_shape = [2, 1]

  output  = [[2,3],[4,5]]             # output_shape = [2, 2]

**Attributes**

* **batch_dims**:
  The number of batch dimensions. The gather of indexing starts from
  dimension of data[batch_dims:] Default value is ``0``.

**Inputs**

* **data** (heterogeneous) - **T**:
  Tensor of rank r >= 1.
* **indices** (heterogeneous) - **tensor(int64)**:
  Tensor of rank q >= 1. All index values are expected to be within
  bounds [-s, s-1] along axis of size s. It is an error if any of the
  index values are out of bounds.

**Outputs**

* **output** (heterogeneous) - **T**:
  Tensor of rank q + r - indices_shape[-1] - 1.

**Type Constraints**

* **T** in (
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input and output types to any tensor type.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td><code>0</code></td><td><code>0</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Given data tensor of rank r >= 1, <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span>indices tensor of rank q >= 1, this operator gathers</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Given data tensor of rank r >= 1, indices tensor of rank q >= 1, <span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">a</span>t<span style="color:#196F3D;">c</span>h<span style="color:#196F3D;">_</span><span style="color:#196F3D;">d</span>i<span style="color:#196F3D;">m</span>s <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span>operator gathers</code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>1</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">slices of data into an output tensor of rank q + r - indices_shape[-1] - 1.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>slices of data into an output tensor of rank q + r - indices_shape[-1] - 1<span style="color:#196F3D;"> </span><span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span>.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">indices is an q-dimensional integer tensor, best thought of as a (q-1)-dimensional tensor of index-tuples into data,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">indices is an q-dimensional integer tensor, best thought of as a (q-1)-dimensional tensor of index-tuples into data,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where each element defines a slice of data</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where each element defines a slice of data</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">batch_dims (denoted as b) is an integer indicating the number of batch dimensions, i.e the leading b number of dimensions of</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">7</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">data tensor and indices are representing the batches, and the gather starts from the b+1 dimension.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Some salient points about the inputs' rank and shape:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Some salient points about the inputs' rank and shape:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks r and q</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks r and q</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">2) The first b dimensions of the shape of indices tensor and data tensor must be equal.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">14</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">15</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">3) b < min(q, r) is to be honored.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">16</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>17</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">2</span>) The indices_shape[-1] should have a value between 1 (inclusive) and rank r (inclusive)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">4</span>) The indices_shape[-1] should have a value between 1 (inclusive) and rank r<span style="color:#196F3D;">-</span><span style="color:#196F3D;">b</span> (inclusive)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>12</code></td><td><code>19</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">3</span>) All values in indices are expected to be within bounds [-s, s-1] along axis of size s (i.e.) -data_shape[i] <= indices[...,i] <= data_shape[i] - 1.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">5</span>) All values in indices are expected to be within bounds [-s, s-1] along axis of size s (i.e.) -data_shape[i] <= indices[...,i] <= data_shape[i] - 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   It is an error if any of the index values are out of bounds.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   It is an error if any of the index values are out of bounds.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output is computed as follows:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output is computed as follows:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output tensor is obtained by mapping each index-tuple in the indices tensor to the corresponding slice of the input data.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output tensor is obtained by mapping each index-tuple in the indices tensor to the corresponding slice of the input data.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>19</code></td><td><code>26</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">1) If indices_shape[-1] > r => error condition</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>1) If indices_shape[-1] > r<span style="color:#196F3D;">-</span><span style="color:#196F3D;">b</span> => error condition</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>21</code></td><td><code>28</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">2) If indices_shape[-1] == r, since the rank of indices is q, indices can be thought of as <span style="color:#BA4A00;">a</span> (q-1)-dimensional tensor</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>2) If indices_shape[-1] == r<span style="color:#196F3D;">-</span><span style="color:#196F3D;">b</span>, since the rank of indices is q, indices can be thought of as <span style="color:#196F3D;">N</span> (q-<span style="color:#196F3D;">b</span><span style="color:#196F3D;">-</span>1)-dimensional tensor<span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td><code>22</code></td><td><code>29</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">   containing 1-D tensors of dimension r<span style="color:#BA4A00;">.</span> <span style="color:#BA4A00;">L</span>et us t<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">k</span><span style="color:#BA4A00;"> </span>o<span style="color:#BA4A00;">f</span> e<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">h</span> <span style="color:#BA4A00;">s</span>uc<span style="color:#BA4A00;">h</span> <span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">r</span>an<span style="color:#BA4A00;">k</span><span style="color:#BA4A00;">e</span>d tens<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span> a<span style="color:#BA4A00;">s</span> <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span>di<span style="color:#BA4A00;">c</span>es<span style="color:#BA4A00;">_</span>s<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>   containing 1-D tensors of dimension r<span style="color:#196F3D;">-</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">,</span> <span style="color:#196F3D;">w</span><span style="color:#196F3D;">h</span>e<span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">N</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>t<span style="color:#196F3D;">e</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span> <span style="color:#196F3D;">e</span><span style="color:#196F3D;">q</span>u<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span>s to <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>e <span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">d</span>uc<span style="color:#196F3D;">t</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span> <span style="color:#196F3D;">1</span><span style="color:#196F3D;"> </span>and <span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span>t<span style="color:#196F3D;">h</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span>n<span style="color:#196F3D;">t</span>s <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span>a<span style="color:#196F3D;">t</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span> di<span style="color:#196F3D;">m</span>e<span style="color:#196F3D;">n</span>s<span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span>s</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">23</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">   Each *scalar value* corresponding to data[indices_slice] is filled into the corresponding location of the (q-1)-dimensional tensor</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>24</code></td><td><code>30</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">   <span style="color:#BA4A00;">t</span>o<span style="color:#BA4A00;"> </span>f<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">m</span> the <span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span>t<span style="color:#BA4A00;">p</span>ut tensor <span style="color:#BA4A00;">(</span>E<span style="color:#BA4A00;">x</span>a<span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">p</span>le <span style="color:#BA4A00;">1</span> bel<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">)</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>   of the <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">L</span><span style="color:#196F3D;">e</span>t<span style="color:#196F3D;"> </span>u<span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span>t<span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">k</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">k</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span>tensor <span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span>Ea<span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span>l<span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span>e<span style="color:#196F3D;">*</span> <span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span> <span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">:</span>b<span style="color:#196F3D;">-</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">s</span>l<span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">]</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">31</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor (Example 1 below)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>26</code></td><td><code>33</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">3) If indices_shape[-1] < r, since the rank of indices is q, indices can be thought of as <span style="color:#BA4A00;">a</span> (q-1)-dimensional tensor</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>3) If indices_shape[-1] < r<span style="color:#196F3D;">-</span><span style="color:#196F3D;">b</span>, since the rank of indices is q, indices can be thought of as <span style="color:#196F3D;">N</span> (q-<span style="color:#196F3D;">b</span><span style="color:#196F3D;">-</span>1)-dimensional tensor</code></td></tr>
    <tr style="1px solid black;"><td><code>27</code></td><td><code>34</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">   containing 1-D tensors of dimension < r. Let us think of each such tensors as indices_slice.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>   containing 1-D tensors of dimension < r<span style="color:#196F3D;">-</span><span style="color:#196F3D;">b</span>. Let us think of each such tensors as indices_slice.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">E</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span></code></td></tr>
    <tr style="1px solid black;"><td><code>28</code></td><td><code>35</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">   <span style="color:#BA4A00;">E</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">*</span>t<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span>o<span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span>d<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span>ata[indices_slice , :] is filled into the corresponding location of the (q-1)-dimensional tensor</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>   to data[<span style="color:#196F3D;">0</span><span style="color:#196F3D;">:</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span>indices_slice , :] is filled into the corresponding location of the (q-<span style="color:#196F3D;">b</span><span style="color:#196F3D;">-</span>1)-dimensional tensor</code></td></tr>
    <tr style="1px solid black;"><td><code>29</code></td><td><code>36</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">   to form the output tensor (Examples 2, 3, and <span style="color:#BA4A00;">4</span> below)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>   to form the output tensor (Examples 2, 3, <span style="color:#196F3D;">4</span><span style="color:#196F3D;"> </span>and <span style="color:#196F3D;">5</span> below)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator is the inverse of ScatterND.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator is the inverse of ScatterND.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">42</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  batch_dims = 0</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">43</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[0,1],[2,3]]   # data_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[0,1],[2,3]]   # data_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [0,3]           # output_shape = [2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [0,3]           # output_shape = [2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">52</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  batch_dims = 0</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">53</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[0,1],[2,3]]  # data_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[0,1],[2,3]]  # data_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[1],[0]]      # indices_shape = [2, 1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[1],[0]]      # indices_shape = [2, 1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[0,1]]  # output_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[0,1]]  # output_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 3</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">62</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  batch_dims = 0</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">63</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 4</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">72</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  batch_dims = 0</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">73</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">80</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">Example 5</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">81</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">82</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  batch_dims = 1</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">83</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">84</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">85</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">86</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  indices = [[1],[0]]             # indices_shape = [2, 1]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">87</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">88</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  output  = [[2,3],[4,5]]             # output_shape = [2, 2]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">89</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">90</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">91</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">92</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **batch_dims**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">93</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The number of batch dimensions. The gather of indexing starts from</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">94</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  dimension of data[batch_dims:] Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">95</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >= 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >= 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **indices** (heterogeneous) - **tensor(int64)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **indices** (heterogeneous) - **tensor(int64)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank q >= 1. All index values are expected to be within</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank q >= 1. All index values are expected to be within</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  bounds [-s, s-1] along axis of size s. It is an error if any of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  bounds [-s, s-1] along axis of size s. It is an error if any of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  index values are out of bounds.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  index values are out of bounds.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank q + r - indices_shape[-1] - 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank q + r - indices_shape[-1] - 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to any tensor type.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to any tensor type.</code></td></tr>
    </table>

.. _l-onnx-op-gathernd-11:

GatherND - 11
=============

**Version**

* **name**: `GatherND (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Given `data` tensor of rank `r` >= 1, and `indices` tensor of rank `q` >= 1, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r` (inclusive)

3) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r` => error condition

2) If `indices_shape[-1] == r`, since the rank of `indices` is `q`, `indices` can be thought of as a `(q-1)`-dimensional tensor
   containing 1-D tensors of dimension `r`. Let us think of each such `r` ranked tensor as `indices_slice`.
   Each *scalar value* corresponding to `data[indices_slice]` is filled into the corresponding location of the `(q-1)`-dimensional tensor
   to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r`, since the rank of `indices` is `q`, `indices` can be thought of as a `(q-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r`. Let us think of each such tensors as `indices_slice`.
   Each *tensor slice* corresponding to `data[indices_slice , :]` is filled into the corresponding location of the `(q-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, and 4 below)

This operator is the inverse of `ScatterND`.

`Example 1`

  data    = [[0,1],[2,3]]   # data_shape = [2, 2]

  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]

  output  = [0,3]           # output_shape = [2]

`Example 2`

  data    = [[0,1],[2,3]]  # data_shape = [2, 2]

  indices = [[1],[0]]      # indices_shape = [2, 1]

  output  = [[2,3],[0,1]]  # output_shape = [2, 2]

`Example 3`

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]

  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]

`Example 4`

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]

  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]

**Inputs**

* **data** (heterogeneous) - **T**:
  Tensor of rank r >= 1.
* **indices** (heterogeneous) - **tensor(int64)**:
  Tensor of rank q >= 1. All index values are expected to be within
  bounds [-s, s-1] along axis of size s. It is an error if any of the
  index values are out of bounds.

**Outputs**

* **output** (heterogeneous) - **T**:
  Tensor of rank q + r - indices_shape[-1] - 1.

**Type Constraints**

* **T** in (
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input and output types to any tensor type.
