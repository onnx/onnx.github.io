
.. _l-onnx-doc-EyeLike:

=======
EyeLike
=======

.. contents::
    :local:


.. _l-onnx-op-eyelike-9:

EyeLike - 9
===========

**Version**

* **name**: `EyeLike (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#EyeLike>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.

**Attributes**

* **dtype**:
  (Optional) The data type for the elements of the output tensor. If
  not specified,the data type of the input tensor T1 is used. If input
  tensor T1 is also notspecified, then type defaults to 'float'.
* **k**:
  (Optional) Index of the diagonal to be populated with ones. Default
  is 0. If T2 is the output, this op sets T2[i, i+k] = 1. k = 0
  populates the main diagonal, k > 0 populates an upper diagonal,  and
  k < 0 populates a lower diagonal. Default value is ``0``.

**Inputs**

* **input** (heterogeneous) - **T1**:
  2D input tensor to copy shape, and optionally, type information
  from.

**Outputs**

* **output** (heterogeneous) - **T2**:
  Output tensor, same shape as input tensor T1.

**Type Constraints**

* **T1** in (
  tensor(bool),
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
  Constrain input types. Strings and complex are not supported.
* **T2** in (
  tensor(bool),
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
  Constrain output types. Strings and complex are not supported.

**Examples**

**without_dtype**

::

    shape = (4, 4)
    node = onnx.helper.make_node(
        'EyeLike',
        inputs=['x'],
        outputs=['y'],
    )

    x = np.random.randint(0, 100, size=shape, dtype=np.int32)
    y = np.eye(shape[0], shape[1], dtype=np.int32)
    expect(node, inputs=[x], outputs=[y], name='test_eyelike_without_dtype')

**with_dtype**

::

    shape = (3, 4)
    node = onnx.helper.make_node(
        'EyeLike',
        inputs=['x'],
        outputs=['y'],
        dtype=onnx.TensorProto.DOUBLE,
    )

    x = np.random.randint(0, 100, size=shape, dtype=np.int32)
    y = np.eye(shape[0], shape[1], dtype=np.float64)
    expect(node, inputs=[x], outputs=[y], name='test_eyelike_with_dtype')

**populate_off_main_diagonal**

::

    shape = (4, 5)
    off_diagonal_offset = 1
    node = onnx.helper.make_node(
        'EyeLike',
        inputs=['x'],
        outputs=['y'],
        k=off_diagonal_offset,
        dtype=onnx.TensorProto.FLOAT,
    )

    x = np.random.randint(0, 100, size=shape, dtype=np.int32)
    y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_eyelike_populate_off_main_diagonal')
