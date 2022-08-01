
.. _l-onnx-doc-Range:

=====
Range
=====

.. contents::
    :local:


.. _l-onnx-op-range-11:

Range - 11
==========

**Version**

* **name**: `Range (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: True
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below-

`number_of_elements = max( ceil( (limit - start) / delta ) , 0 )`

The pseudocode determining the contents of the output is shown below-

`for(int i=0; i<number_of_elements; ++i)`

`{`

`    output[i] =  start + (i * delta);  `

`}`

`Example 1`
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]

`Example 2`
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]

**Inputs**

* **start** (heterogeneous) - **T**:
  Scalar. First entry for the range of output values.
* **limit** (heterogeneous) - **T**:
  Scalar. Exclusive upper limit for the range of output values.
* **delta** (heterogeneous) - **T**:
  Scalar. Value to step by.

**Outputs**

* **output** (heterogeneous) - **T**:
  A 1-D tensor with same type as the inputs containing generated range
  of values.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int16),
  tensor(int32),
  tensor(int64)
  ):
  Constrain input types to common numeric type tensors.

**Examples**
