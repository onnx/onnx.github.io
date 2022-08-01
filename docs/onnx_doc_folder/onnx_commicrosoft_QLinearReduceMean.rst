
.. _l-onnx-doccom.microsoft-QLinearReduceMean:

=================================
com.microsoft - QLinearReduceMean
=================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qlinearreducemean-1:

QLinearReduceMean - 1 (com.microsoft)
=====================================

**Version**

* **name**: `QLinearReduceMean (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QLinearReduceMean>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Computes the mean of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.
Input and Output scales and zero points are used to requantize the output in a new range.
This helps to improve accuracy as after ReduceMean operation the range of the output is expected to decrease.

::

    "Output = Dequantize(Input) -> ReduceMean on fp32 data -> Quantize(output)",

**Attributes**

* **axes** (required):
  A list of integers, along which to reduce. The default is to reduce
  over all the dimensions of the input tensor. Default value is ``?``.
* **keepdims** (required):
  Keep the reduced dimension or not, default 1 mean keep reduced
  dimension. Default value is ``?``.

**Inputs**

Between 4 and 5 inputs.

* **data** (heterogeneous) - **T**:
  An input tensor.
* **data_scale** (heterogeneous) - **tensor(float)**:
  Input scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **data_zero_point** (optional, heterogeneous) - **T**:
  Input zero point. Default value is 0 if it's not specified. It's a
  scalar, which means a per-tensor/layer quantization.
* **reduced_scale** (heterogeneous) - **tensor(float)**:
  Output scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **reduced_zero_point** (optional, heterogeneous) - **T**:
  Output zero point. Default value is 0 if it's not specified. It's a
  scalar, which means a per-tensor/layer quantization.

**Outputs**

* **reduced** (heterogeneous) - **T**:
  Reduced output tensor.

**Examples**
