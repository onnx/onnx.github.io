
.. _l-onnx-doccom.microsoft-CDist:

=====================
com.microsoft - CDist
=====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-cdist-1:

CDist - 1 (com.microsoft)
=========================

**Version**

* **name**: `CDist (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.CDist>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **metric**:
  The distance metric to use. If a string, the distance function can
  be "braycurtis", "canberra", "chebyshev", "cityblock",
  "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard",
  "jensenshannon", "kulsinski", "mahalanobis", "matching",
  "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
  "sokalmichener", "sokalsneath", "sqeuclidean", "wminkowski", "yule". Default value is ``?``.

**Inputs**

* **A** (heterogeneous) - **T**:
  2D matrix with shape (M,N)
* **B** (heterogeneous) - **T**:
  2D matrix with shape (K,N)

**Outputs**

* **C** (heterogeneous) - **T**:
  A 2D Matrix that represents the distance between each pair of the
  two collections of inputs.

**Examples**
