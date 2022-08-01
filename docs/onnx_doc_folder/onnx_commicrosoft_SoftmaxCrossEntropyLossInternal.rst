
.. _l-onnx-doccom.microsoft-SoftmaxCrossEntropyLossInternal:

===============================================
com.microsoft - SoftmaxCrossEntropyLossInternal
===============================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-softmaxcrossentropylossinternal-1:

SoftmaxCrossEntropyLossInternal - 1 (com.microsoft)
===================================================

**Version**

* **name**: `SoftmaxCrossEntropyLossInternal (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SoftmaxCrossEntropyLossInternal>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SoftmaxCrossEntropyLossInternal

**Attributes**

* **reduction**:
  Type of reduction to apply to loss: none, sum, mean(default).
  'none': the output is the loss for each sample in the batch.'sum':
  the output will be summed. 'mean': the sum of the output will be
  divided by the batch_size. Default value is ``?``.

**Inputs**

Between 2 and 4 inputs.

* **scores** (heterogeneous) - **T**:
  The predicted outputs with shape [batch_size, class_size], or
  [batch_size, class_size, D1, D2 , ..., Dk], where K is the number of
  dimensions.
* **labels** (heterogeneous) - **Tind**:
  The ground truth output tensor, with shape [batch_size], or
  [batch_size, D1, D2, ..., Dk], where K is the number of dimensions.
  Labels element value shall be in range of [0, C). If ignore_index is
  specified, it may have a value outside [0, C) and the label values
  should either be in the range [0, C) or have the value ignore_index.
* **weights** (optional, heterogeneous) - **T**:
  A manual rescaling weight given to each class. If given, it has to
  be a 1D Tensor assigning weight to each of the classes. Otherwise,
  it is treated as if having all ones.
* **ignore_index** (optional, heterogeneous) - **I**:
  Scalar tensor to specify a target value that is ignored and does not
  contribute to the input gradient.

**Outputs**

* **output** (heterogeneous) - **T**:
  Weighted loss float Tensor. If reduction is 'none', this has the
  shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of
  K-dimensional loss. Otherwise, it is a scalar.
* **log_prob** (heterogeneous) - **T**:
  Log probability tensor. If the output of softmax is prob, its value
  is log(prob).

**Examples**
