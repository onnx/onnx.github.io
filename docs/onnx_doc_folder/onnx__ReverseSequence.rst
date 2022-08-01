
.. _l-onnx-doc-ReverseSequence:

===============
ReverseSequence
===============

.. contents::
    :local:


.. _l-onnx-op-reversesequence-10:

ReverseSequence - 10
====================

**Version**

* **name**: `ReverseSequence (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReverseSequence>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]

**Attributes**

* **batch_axis**:
  (Optional) Specify which axis is batch axis. Must be one of 1
  (default), or 0. Default value is ``1``.
* **time_axis**:
  (Optional) Specify which axis is time axis. Must be one of 0
  (default), or 1. Default value is ``0``.

**Inputs**

* **input** (heterogeneous) - **T**:
  Tensor of rank r >= 2.
* **sequence_lens** (heterogeneous) - **tensor(int64)**:
  Tensor specifying lengths of the sequences in a batch. It has shape
  `[batch_size]`.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Tensor with same shape of input.

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
  Input and output types can be of any tensor type.

**Examples**

**reversesequence_time**

::

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x', 'sequence_lens'],
        outputs=['y'],
        time_axis=0,
        batch_axis=1,
    )
    x = np.array([[0.0, 4.0, 8.0, 12.0],
                  [1.0, 5.0, 9.0, 13.0],
                  [2.0, 6.0, 10.0, 14.0],
                  [3.0, 7.0, 11.0, 15.0]], dtype=np.float32)
    sequence_lens = np.array([4, 3, 2, 1], dtype=np.int64)

    y = np.array([[3.0, 6.0, 9.0, 12.0],
                  [2.0, 5.0, 8.0, 13.0],
                  [1.0, 4.0, 10.0, 14.0],
                  [0.0, 7.0, 11.0, 15.0]], dtype=np.float32)

    expect(node, inputs=[x, sequence_lens], outputs=[y],
           name='test_reversesequence_time')

**reversesequence_batch**

::

    node = onnx.helper.make_node(
        'ReverseSequence',
        inputs=['x', 'sequence_lens'],
        outputs=['y'],
        time_axis=1,
        batch_axis=0,
    )
    x = np.array([[0.0, 1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0, 7.0],
                  [8.0, 9.0, 10.0, 11.0],
                  [12.0, 13.0, 14.0, 15.0]], dtype=np.float32)
    sequence_lens = np.array([1, 2, 3, 4], dtype=np.int64)

    y = np.array([[0.0, 1.0, 2.0, 3.0],
                  [5.0, 4.0, 6.0, 7.0],
                  [10.0, 9.0, 8.0, 11.0],
                  [15.0, 14.0, 13.0, 12.0]], dtype=np.float32)

    expect(node, inputs=[x, sequence_lens], outputs=[y],
           name='test_reversesequence_batch')
