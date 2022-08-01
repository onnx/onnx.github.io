
.. _l-onnx-doccom.microsoft-Unique:

======================
com.microsoft - Unique
======================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-unique-1:

Unique - 1 (com.microsoft)
==========================

**Version**

* **name**: `Unique (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Unique>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Finds all the unique values (deduped list) present in the given input tensor.
This operator returns 3 outputs.
The first output tensor 'uniques' contains all of the unique elements of the input,
sorted in the same order that they occur in the input.
The second output tensor 'idx' is the same size as the input and it contains the index
of each value of the input in 'uniques'.
The third output tensor 'counts' contains the count of each element of 'uniques' in the input.
Example:
  input_x = [2, 1, 1, 3, 4, 3]
  output_uniques = [2, 1, 3, 4]
  output_idx = [0, 1, 1, 2, 3, 2]
  output_counts = [1, 2, 2, 1]

**Inputs**

* **x** (heterogeneous) - **T**:
  A 1-D input tensor that is to be processed.

**Outputs**

* **y** (heterogeneous) - **T**:
  A 1-D tensor of the same type as 'x' containing all the unique
  values in 'x' sorted in the same order that they occur in the input
  'x'
* **idx** (heterogeneous) - **tensor(int64)**:
  A 1-D INT64 tensor of the same size as 'x' containing the indices
  for each value in 'x' in the output 'uniques'
* **counts** (heterogeneous) - **tensor(int64)**:
  A 1-D INT64 tensor containing the the count of each element of
  'uniques' in the input 'x'

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
