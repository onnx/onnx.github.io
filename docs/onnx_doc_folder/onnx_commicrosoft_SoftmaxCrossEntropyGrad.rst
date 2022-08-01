
.. _l-onnx-doccom.microsoft-SoftmaxCrossEntropyGrad:

=======================================
com.microsoft - SoftmaxCrossEntropyGrad
=======================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-softmaxcrossentropygrad-1:

SoftmaxCrossEntropyGrad - 1 (com.microsoft)
===========================================

**Version**

* **name**: `SoftmaxCrossEntropyGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SoftmaxCrossEntropyGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SoftmaxCrossEntropyGrad

**Attributes**

* **reduction**:
  Type of reduction to apply to loss: none, sum, mean(default).
  'none': the output is the loss for each sample in the batch.'sum':
  the output will be summed. 'mean': the sum of the output will be
  divided by the batch_size. Default value is ``?``.

**Inputs**

* **dY** (heterogeneous) - **T**:
  gradient of Y
* **log_prob** (heterogeneous) - **T**:
  logsoftmax(logits), N-D input of shape (-1, num_classes).
* **label** (heterogeneous) - **T**:
  The onehot label is N-D input with the same shape as logits.

**Outputs**

* **d_logits** (heterogeneous) - **T**:
  gradient of logits

**Examples**
