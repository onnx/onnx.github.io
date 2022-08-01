
.. _l-onnx-doccom.microsoft-GatherND:

========================
com.microsoft - GatherND
========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-gathernd-1:

GatherND - 1 (com.microsoft)
============================

**Version**

* **name**: `GatherND (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.GatherND>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Given `data` tensor of rank r >= 1, and `indices` tensor of rank q >= 1, gather
slices of `data` into an output tensor of rank q - 1 + r - indices[-1].
Example 1:
  data    = [[0,1],[2,3]]
  indices = [[0,0],[1,1]]
  output  = [0,3]
Example 2:
  data    = [[0,1],[2,3]]
  indices = [[1],[0]]
  output  = [[2,3],[0,1]]
Example 3:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[0,1],[1,0]]
  output  = [[2,3],[4,5]]
Example 4:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[[0,1]],[[1,0]]]
  output  = [[[2,3]],[[4,5]]]

**Inputs**

* **data** (heterogeneous) - **T**:
  Tensor of rank r >= 1.
* **indices** (heterogeneous) - **Tind**:
  Tensor of rank q >= 1.

**Outputs**

* **output** (heterogeneous) - **T**:
  Tensor of rank q-1+r-indices[-1].

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
