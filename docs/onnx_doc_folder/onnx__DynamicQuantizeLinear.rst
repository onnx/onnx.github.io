
.. _l-onnx-doc-DynamicQuantizeLinear:

=====================
DynamicQuantizeLinear
=====================

.. contents::
    :local:


.. _l-onnx-op-dynamicquantizelinear-11:

DynamicQuantizeLinear - 11
==========================

**Version**

* **name**: `DynamicQuantizeLinear (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#DynamicQuantizeLinear>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: True
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

A Function to fuse calculation for Scale, Zero Point and FP32->8Bit convertion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
::

     y_scale = (max(x) - min(x))/(qmax - qmin)
     * where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
     * data range is adjusted to include 0.

Zero point is calculated as:
::

    intermediate_zero_point = qmin - min(x)/y_scale
    y_zero_point = cast(round(saturate(itermediate_zero_point)))
    * where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
    * for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
    * rounding to nearest ties to even.

Data quantization formula is:
::

    y = saturate (round (x / y_scale) + y_zero_point)
    * for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
    * rounding to nearest ties to even.

**Inputs**

* **x** (heterogeneous) - **T1**:
  Input tensor

**Outputs**

* **y** (heterogeneous) - **T2**:
  Quantized output tensor
* **y_scale** (heterogeneous) - **tensor(float)**:
  Output scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **y_zero_point** (heterogeneous) - **T2**:
  Output zero point. It's a scalar, which means a per-tensor/layer
  quantization.

**Type Constraints**

* **T1** in (
  tensor(float)
  ):
  Constrain 'x' to float tensor.
* **T2** in (
  tensor(uint8)
  ):
  Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.

**Examples**
