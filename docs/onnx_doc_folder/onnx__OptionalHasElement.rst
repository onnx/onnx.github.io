
.. _l-onnx-doc-OptionalHasElement:

==================
OptionalHasElement
==================

.. contents::
    :local:


.. _l-onnx-op-optionalhaselement-15:

OptionalHasElement - 15
=======================

**Version**

* **name**: `OptionalHasElement (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#OptionalHasElement>`_
* **domain**: **main**
* **since_version**: **15**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 15**.

**Summary**

Returns true if the optional-type input contains an element. If it is an empty optional-type, this op returns false.

**Inputs**

* **input** (heterogeneous) - **O**:
  The optional input.

**Outputs**

* **output** (heterogeneous) - **B**:
  A scalar boolean tensor. If true, it indicates that optional-type
  input contains an element. Otherwise, it is empty.

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
* **B** in (
  tensor(bool)
  ):
  Constrain output to a boolean tensor.

**Examples**

**empty**

::

    optional = None
    tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.INT32, shape=[])
    input_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)
    node = onnx.helper.make_node(
        'OptionalHasElement',
        inputs=['optional_input'],
        outputs=['output']
    )
    output = optional_has_element_reference_implementation(optional)
    expect(node, inputs=[optional], outputs=[output],
           input_type_protos=[input_type_proto],
           name='test_optional_has_element_empty')
