
.. _l-onnx-doc-Unique:

======
Unique
======

.. contents::
    :local:


.. _l-onnx-op-unique-11:

Unique - 11
===========

**Version**

* **name**: `Unique (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unique>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
The first output tensor 'Y' contains all unique values or subtensors of the input.
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurance in 'X'..
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'. ".
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

Example 1:
  input_X = [2, 1, 1, 3, 4, 3]
  attribute_sorted = 0
  attribute_axis = None
  output_Y = [2, 1, 3, 4]
  output_indices = [0, 1, 3, 4]
  output_inverse_indices = [0, 1, 1, 2, 3, 2]
  output_counts = [1, 2, 2, 1]

Example 2:
  input_X = [[1, 3], [2, 3]]
  attribute_sorted = 1
  attribute_axis = None
  output_Y = [1, 2, 3]
  output_indices = [0, 2, 1]
  output_inverse_indices = [0, 2, 1, 2]
  output_counts = [1, 1, 2]

Example 3:
  input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
  attribute_sorted = 1
  attribute_axis = 0
  output_Y = [[1, 0, 0], [2, 3, 4]]
  output_indices = [0, 2]
  output_inverse_indices = [0, 0, 1]
  output_counts = [2, 1]

Example 4:
  input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
             [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
  attribute_sorted = 1
  attribute_axis = 1

  intermediate data are presented below for better understanding:

  there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
  A: [[1, 1], [1, 1]],
     [[0, 1], [0, 1]],
     [[2, 1], [2, 1]],
     [[0, 1], [0, 1]].

  there are 3 unique subtensors:
  [[1, 1], [1, 1]],
  [[0, 1], [0, 1]],
  [[2, 1], [2, 1]].

  sorted unique subtensors:
  B: [[0, 1], [0, 1]],
     [[1, 1], [1, 1]],
     [[2, 1], [2, 1]].

  output_Y is constructed from B:
  [[[0. 1.], [1. 1.], [2. 1.]],
   [[0. 1.], [1. 1.], [2. 1.]]]

  output_indices is to map from B to A:
  [1, 0, 2]

  output_inverse_indices is to map from A to B:
  [1, 0, 2, 0]

  output_counts = [2 1 1]

**Attributes**

* **axis**:
  (Optional) The dimension to apply unique. If not specified, the
  unique elements of the flattened input are returned. Negative value
  means counting dimensions from the back. Accepted range is [-r, r-1]
  where r = rank(input).
* **sorted**:
  (Optional) Whether to sort the unique elements in ascending order
  before returning as output. Must be one of 0, or 1 (default). Default value is ``1``.

**Inputs**

* **X** (heterogeneous) - **T**:
  A N-D input tensor that is to be processed.

**Outputs**

Between 1 and 4 outputs.

* **Y** (heterogeneous) - **T**:
  A tensor of the same type as 'X' containing all the unique values or
  subtensors sliced along a provided 'axis' in 'X', either sorted or
  maintained in the same order they occur in input 'X'
* **indices** (optional, heterogeneous) - **tensor(int64)**:
  A 1-D INT64 tensor containing indices of 'Y' elements' first
  occurance in 'X'. When 'axis' is provided, it contains indices to
  subtensors in input 'X' on the 'axis'. When 'axis' is not provided,
  it contains indices to values in the flattened input tensor.
* **inverse_indices** (optional, heterogeneous) - **tensor(int64)**:
  A 1-D INT64 tensor containing, for elements of 'X', its
  corresponding indices in 'Y'. When 'axis' is provided, it contains
  indices to subtensors in output 'Y' on the 'axis'. When 'axis' is
  not provided, it contains indices to values in output 'Y'.
* **counts** (optional, heterogeneous) - **tensor(int64)**:
  A 1-D INT64 tensor containing the count of each element of 'Y' in
  input 'X'

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
  Input can be of any tensor type.

**Examples**

**sorted_without_axis**

::

    node_sorted = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts']
    )

    x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
    y, indices, inverse_indices, counts = np.unique(x, True, True, True)
    indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
    expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_without_axis')

**not_sorted_without_axis**

::

    node_not_sorted = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts'],
        sorted=0
    )
    # numpy unique does not retain original order (it sorts the output unique values)
    # https://github.com/numpy/numpy/issues/8621
    # we need to recover unsorted output and indices
    x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
    y, indices, inverse_indices, counts = np.unique(x, True, True, True)

    # prepare index mapping from sorted to unsorted
    argsorted_indices = np.argsort(indices)
    inverse_indices_map = {i: si for i, si in zip(argsorted_indices, np.arange(len(argsorted_indices)))}

    indices = indices[argsorted_indices]
    y = np.take(x, indices, axis=0)
    inverse_indices = np.asarray([inverse_indices_map[i] for i in inverse_indices], dtype=np.int64)
    counts = counts[argsorted_indices]
    indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
    # print(y)
    # [2.0, 1.0, 3.0, 4.0]
    # print(indices)
    # [0 1 3 4]
    # print(inverse_indices)
    # [0, 1, 1, 2, 3, 2]
    # print(counts)
    # [1, 2, 2, 1]

    expect(node_not_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_not_sorted_without_axis')

**sorted_with_axis**

::

    node_sorted = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts'],
        sorted=1,
        axis=0
    )

    x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]], dtype=np.float32)
    y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=0)
    indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
    # print(y)
    # [[1. 0. 0.]
    #  [2. 3. 4.]]
    # print(indices)
    # [0 2]
    # print(inverse_indices)
    # [0 0 1]
    # print(counts)
    # [2 1]

    expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_with_axis')

**sorted_with_axis_3d**

::

    node_sorted = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts'],
        sorted=1,
        axis=1
    )

    x = np.array([[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
                  [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]], dtype=np.float32)
    y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=1)
    indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
    # print(y)
    # [[[0. 1.]
    #  [1. 1.]
    #  [2. 1.]]
    # [[0. 1.]
    #  [1. 1.]
    #  [2. 1.]]]
    # print(indices)
    # [1 0 2]
    # print(inverse_indices)
    # [1 0 2 0]
    # print(counts)
    # [2 1 1]
    expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_with_axis_3d')

**sorted_with_negative_axis**

::

    node_sorted = onnx.helper.make_node(
        'Unique',
        inputs=['X'],
        outputs=['Y', 'indices', 'inverse_indices', 'counts'],
        sorted=1,
        axis=-1
    )

    x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 3]], dtype=np.float32)
    y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=-1)
    indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
    # print(y)
    # [[0. 1.]
    #  [0. 1.]
    #  [3. 2.]]
    # print(indices)
    # [1 0]
    # print(inverse_indices)
    # [1 0 0]
    # print(counts)
    # [2 1]

    expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_with_negative_axis')
