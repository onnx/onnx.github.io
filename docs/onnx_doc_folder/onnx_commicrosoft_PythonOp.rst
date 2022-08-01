
.. _l-onnx-doccom.microsoft-PythonOp:

========================
com.microsoft - PythonOp
========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-pythonop-1:

PythonOp - 1 (com.microsoft)
============================

**Version**

* **name**: `PythonOp (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.PythonOp>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Wrapper of Pytorch's autograd.Function implementation.

**Attributes**

* **inplace**:
  Indicate if the output should reuse input memory. Default value is ``?``.
* **input_convention** (required):
  input_convention[i]==c means a non-tensor argument.
  input_convention[i]==d means a tensor. Default value is ``?``.
* **input_float_scalar_positions**:
 Default value is ``?``.
* **input_float_scalars**:
  Python float arguments. Default value is ``?``.
* **input_float_tuple_begins**:
 Default value is ``?``.
* **input_float_tuple_positions**:
 Default value is ``?``.
* **input_float_tuples**:
 Default value is ``?``.
* **input_int_scalar_positions**:
 Default value is ``?``.
* **input_int_scalars**:
  Python int arguments. Default value is ``?``.
* **input_int_tuple_begins**:
 Default value is ``?``.
* **input_int_tuple_positions**:
 Default value is ``?``.
* **input_int_tuples**:
  Python int-tuple arguments. Default value is ``?``.
* **input_pointer_scalar_positions**:
 Default value is ``?``.
* **input_pointer_scalars**:
 Default value is ``?``.
* **input_requires_grads** (required):
  Flags to indicate whether the torch.autograd.apply's inputs require
  gradients (including flags for both tensor and non-tensor inputs) Default value is ``?``.
* **input_tensor_ranks** (required):
  Input tensors' ranks of autograd.Function.apply. Default value is ``?``.
* **input_tensor_types** (required):
  Input types of autograd.Function.apply. Default value is ``?``.
* **name** (required):
  Name of custom class. Default value is ``?``.
* **output_tensor_ranks** (required):
  Output tensors' ranks of autograd.Function.apply. Default value is ``?``.
* **output_tensor_requires_grads** (required):
  Flags to indicate which output has gradient Default value is ``?``.
* **output_tensor_types** (required):
  Output types of autograd.Function.apply. Default value is ``?``.
* **training_mode**:
  Indicate if the model is exported in training_mode, by default,
  False. Default value is ``?``.

**Inputs**

Between 1 and 2147483647 inputs.

* **inputs** (variadic) - **T**:
  Module outputs to be returned to pytorch.

**Outputs**

Between 2 and 2147483647 outputs.

* **context** (heterogeneous) - **TInt64**:
  Address of context created in this operator. It can be used in
  backward.
* **outputs** (variadic) - **T**:
  Outputs returned from pytorch.

**Examples**
