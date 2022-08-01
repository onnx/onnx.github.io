
.. _l-onnx-doc-ConstantOfShape:

===============
ConstantOfShape
===============

.. contents::
    :local:


.. _l-onnx-op-constantofshape-9:

ConstantOfShape - 9
===================

**Version**

* **name**: `ConstantOfShape (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

Generate a tensor with given value and shape.

**Attributes**

* **value**:
  (Optional) The value of the output elements.Should be a one-element
  tensor. If not specified, it defaults to a tensor of value 0 and
  datatype float32

**Inputs**

* **input** (heterogeneous) - **T1**:
  1D tensor. The shape of the expected output tensor. If empty tensor
  is given, the output would be a scalar. All values must be >= 0.

**Outputs**

* **output** (heterogeneous) - **T2**:
  Output tensor of shape specified by 'input'.If attribute 'value' is
  specified, the value and datatype of the output tensor is taken from
  'value'.If attribute 'value' is not specified, the value in the
  output defaults to 0, and the datatype defaults to float32.

**Type Constraints**

* **T1** in (
  tensor(int64)
  ):
  Constrain input types.
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
  Constrain output types to be numerics.

**Examples**

**float_ones**

::

    x = np.array([4, 3, 2]).astype(np.int64)
    tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT,
                                           [1], [1])
    node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['x'],
        outputs=['y'],
        value=tensor_value,
    )

    y = np.ones(x, dtype=np.float32)
    expect(node, inputs=[x], outputs=[y],
           name='test_constantofshape_float_ones')

**int32_zeros**

::

    x = np.array([10, 6]).astype(np.int64)
    tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.INT32,
                                           [1], [0])
    node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['x'],
        outputs=['y'],
        value=tensor_value,
    )
    y = np.zeros(x, dtype=np.int32)
    expect(node, inputs=[x], outputs=[y],
           name='test_constantofshape_int_zeros')

**int32_shape_zero**

::

    x = np.array([0, ]).astype(np.int64)
    tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.INT32,
                                           [1], [0])
    node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['x'],
        outputs=['y'],
        value=tensor_value,
    )
    y = np.zeros(x, dtype=np.int32)
    expect(node, inputs=[x], outputs=[y],
           name='test_constantofshape_int_shape_zero')
