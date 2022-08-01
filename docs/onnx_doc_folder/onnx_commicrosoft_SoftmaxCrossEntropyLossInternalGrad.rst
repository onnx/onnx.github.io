
.. _l-onnx-doccom.microsoft-SoftmaxCrossEntropyLossInternalGrad:

===================================================
com.microsoft - SoftmaxCrossEntropyLossInternalGrad
===================================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-softmaxcrossentropylossinternalgrad-1:

SoftmaxCrossEntropyLossInternalGrad - 1 (com.microsoft)
=======================================================

**Version**

* **name**: `SoftmaxCrossEntropyLossInternalGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SoftmaxCrossEntropyLossInternalGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SoftmaxCrossEntropyLossInternalGrad

**Attributes**

* **reduction**:
  Type of reduction to apply to loss: none, sum, mean(default).
  'none': the output is the loss for each sample in the batch.'sum':
  the output will be summed. 'mean': the sum of the output will be
  divided by the batch_size. Default value is ``?``.

**Inputs**

Between 3 and 5 inputs.

* **dY** (heterogeneous) - **T**:
  gradient of Y
* **log_prob** (heterogeneous) - **T**:
  logsoftmax(logits), (N+1)-D input of shape (batch_size).
* **label** (heterogeneous) - **Tind**:
  label is N-D input whose shape should match that of logits. It is a
  tensor of nonnegative integers, where each element is the
  nonnegative integer label for the element of the batch.
* **weight** (optional, heterogeneous) - **T**:
  weight for each sample. The shape is 1-D tensor.
* **ignore_index** (optional, heterogeneous) - **I**:
  Scalar tensor to specify a target value that is ignored and does not
  contribute to the input gradient.

**Outputs**

* **d_logits** (heterogeneous) - **T**:
  gradient of logits

**Examples**
