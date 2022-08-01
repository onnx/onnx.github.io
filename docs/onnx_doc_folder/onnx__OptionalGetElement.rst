
.. _l-onnx-doc-OptionalGetElement:

==================
OptionalGetElement
==================

.. contents::
    :local:


.. _l-onnx-op-optionalgetelement-15:

OptionalGetElement - 15
=======================

**Version**

* **name**: `OptionalGetElement (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#OptionalGetElement>`_
* **domain**: **main**
* **since_version**: **15**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 15**.

**Summary**

Outputs the element in the optional-type input. It is an error if the input value does not have an element
and the behavior is undefined in this case.

**Inputs**

* **input** (heterogeneous) - **O**:
  The optional input.

**Outputs**

* **output** (heterogeneous) - **V**:
  Output element in the optional input.

**Type Constraints**

* **O** in (
  optional(seq(tensor(bool))),
  optional(seq(tensor(complex128))),
  optional(seq(tensor(complex64))),
  optional(seq(tensor(double))),
  optional(seq(tensor(float))),
  optional(seq(tensor(float16))),
  optional(seq(tensor(int16))),
  optional(seq(tensor(int32))),
  optional(seq(tensor(int64))),
  optional(seq(tensor(int8))),
  optional(seq(tensor(string))),
  optional(seq(tensor(uint16))),
  optional(seq(tensor(uint32))),
  optional(seq(tensor(uint64))),
  optional(seq(tensor(uint8))),
  optional(tensor(bool)),
  optional(tensor(complex128)),
  optional(tensor(complex64)),
  optional(tensor(double)),
  optional(tensor(float)),
  optional(tensor(float16)),
  optional(tensor(int16)),
  optional(tensor(int32)),
  optional(tensor(int64)),
  optional(tensor(int8)),
  optional(tensor(string)),
  optional(tensor(uint16)),
  optional(tensor(uint32)),
  optional(tensor(uint64)),
  optional(tensor(uint8))
  ):
  Constrain input type to optional tensor and optional sequence types.
* **V** in (
  seq(tensor(bool)),
  seq(tensor(complex128)),
  seq(tensor(complex64)),
  seq(tensor(double)),
  seq(tensor(float)),
  seq(tensor(float16)),
  seq(tensor(int16)),
  seq(tensor(int32)),
  seq(tensor(int64)),
  seq(tensor(int8)),
  seq(tensor(string)),
  seq(tensor(uint16)),
  seq(tensor(uint32)),
  seq(tensor(uint64)),
  seq(tensor(uint8)),
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
  Constrain output type to all tensor or sequence types.

**Examples**

**get_element_tensor**

::

    optional = np.array([1, 2, 3, 4]).astype(np.float32)
    tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.FLOAT, shape=[4, ])
    input_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)

    node = onnx.helper.make_node(
        'OptionalGetElement',
        inputs=['optional_input'],
        outputs=['output']
    )
    output = optional_get_element_reference_implementation(optional)
    expect(node, inputs=[optional], outputs=[output],
           input_type_protos=[input_type_proto],
           name='test_optional_get_element')

**get_element_sequence**

::

    optional = [np.array([1, 2, 3, 4]).astype(np.int32)]
    tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.INT32, shape=[4, ])
    seq_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
    input_type_proto = onnx.helper.make_optional_type_proto(seq_type_proto)

    node = onnx.helper.make_node(
        'OptionalGetElement',
        inputs=['optional_input'],
        outputs=['output']
    )
    output = optional_get_element_reference_implementation(optional)
    expect(node, inputs=[optional], outputs=[output],
           input_type_protos=[input_type_proto],
           name='test_optional_get_element_sequence')
