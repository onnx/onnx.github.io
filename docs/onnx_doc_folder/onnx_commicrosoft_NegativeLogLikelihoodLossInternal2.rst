
.. _l-onnx-doccom.microsoft-NegativeLogLikelihoodLossInternal2:

==================================================
com.microsoft - NegativeLogLikelihoodLossInternal2
==================================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-negativeloglikelihoodlossinternal2-1:

NegativeLogLikelihoodLossInternal2 - 1 (com.microsoft)
======================================================

**Version**

* **name**: `NegativeLogLikelihoodLossInternal2 (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.NegativeLogLikelihoodLossInternal2>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

NegativeLogLikelihoodLossInternal

**Attributes**

* **reduction**:
  Type of reduction to apply to loss: none, sum, mean(default).
  'none': the output is the loss for each sample in the batch.'sum':
  the output will be summed. 'mean': the sum of the output will be
  divided by the batch_size. Default value is ``?``.

**Inputs**

Between 2 and 4 inputs.

* **input** (heterogeneous) - **T**:
  Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).
* **target** (heterogeneous) - **Tind**:
  Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element
  value shall be in range of [0, C). If ignore_index is specified, it
  may have a value outside [0, C) and the target values should either
  be in the range [0, C) or have the value ignore_index.
* **weight** (optional, heterogeneous) - **T**:
  Optional rescaling weight tensor. If given, it has to be a tensor of
  size C. Otherwise, it is treated as if having all ones.
* **ignore_index** (optional, heterogeneous) - **I**:
  Scalar tensor to specify a target value that is ignored and does not
  contribute to the input gradient.

**Outputs**

* **loss** (heterogeneous) - **T**:
  The negative log likelihood loss

**Examples**
