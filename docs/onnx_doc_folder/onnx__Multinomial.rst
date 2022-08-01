
.. _l-onnx-doc-Multinomial:

===========
Multinomial
===========

.. contents::
    :local:


.. _l-onnx-op-multinomial-7:

Multinomial - 7
===============

**Version**

* **name**: `Multinomial (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Multinomial>`_
* **domain**: **main**
* **since_version**: **7**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 7**.

**Summary**

Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.

**Attributes**

* **dtype**:
  (Optional) The data type for the elements of the output tensor, if
  not specified, we will use int32. Default value is ``6``.
* **sample_size**:
  Number of times to sample. Default value is ``1``.
* **seed**:
  (Optional) Seed to the random generator, if not specified we will
  auto generate one.

**Inputs**

* **input** (heterogeneous) - **T1**:
  Input tensor with shape [batch_size, class_size], where class_size
  is the number of all possible outcomes. Each value along the axis
  zero represents the unnormalized log-probability of each
  corresponding outcome in a batch.

**Outputs**

* **output** (heterogeneous) - **T2**:
  Output tensor with shape [batch_size, sample_size], where
  sample_size is the number of times to sample. Each value along the
  axis zero represents the outcome of the corresponding sample in a
  batch.

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input types to float tensors.
* **T2** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain output types to integral tensors.

**Examples**
