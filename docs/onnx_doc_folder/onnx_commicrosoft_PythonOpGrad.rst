
.. _l-onnx-doccom.microsoft-PythonOpGrad:

============================
com.microsoft - PythonOpGrad
============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-pythonopgrad-1:

PythonOpGrad - 1 (com.microsoft)
================================

**Version**

* **name**: `PythonOpGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.PythonOpGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Wrapper of Pytorch's autograd.Function's backward implementaiton.

**Attributes**

* **inplace**:
  Indicate if the output should reuse input memory. Todo(pengwa): do
  we really need it? Default value is ``?``.
* **input_tensor_ranks**:
  Input ranks of autograd.Function.backward (including only tensor
  inputs).This attribute is mostly used for input checks for better
  robustness. Default value is ``?``.
* **input_tensor_requires_grads** (required):
  Flags to indicate which inputs have gradients (including only tensor
  inputs).This attribute is mostly used for input checks for better
  robustness. Default value is ``?``.
* **input_tensor_types**:
  Input types of autograd.Function.backward (including only tensor
  inputs).This attribute is mostly used for input checks for better
  robustnes. Default value is ``?``.
* **name** (required):
  Name of custom class. Default value is ``?``.
* **output_convention** (required):
  A string inidicating autograd.Function.backward outputs's type.value
  'c' - non-tensor output; value 'd' - tensor output. Default value is ``?``.
* **output_tensor_ranks**:
  Output ranks of autograd.Function.backward outputs (including only
  tensor outputs). Default value is ``?``.
* **output_tensor_requires_grads** (required):
  Flags to indicate which outputs have gradients (including only
  tensor outputs). Default value is ``?``.
* **output_tensor_types**:
  Output types of autograd.Function.backward outputs (including only
  tensor outputs). Default value is ``?``.

**Inputs**

Between 2 and 2147483647 inputs.

* **context** (heterogeneous) - **TInt64**:
  Address of context created in this operator. It should be generated
  by the corresponding forward.
* **inputs** (variadic) - **T**:
  There are 2*N inputs:   N gradient inputs (as inputs of
  autograd.Function.backward) +   N forward run activations of
  autograd.Function.apply.The N forward run inputs are used as control
  dependency between PythonOpGrad and PythonOp

**Outputs**

Between 1 and 2147483647 outputs.

* **outputs** (variadic) - **T**:
  Outputs returned from pytorch.

**Examples**
