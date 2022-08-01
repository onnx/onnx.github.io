
.. _l-onnx-doccom.microsoft-BifurcationDetector:

===================================
com.microsoft - BifurcationDetector
===================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-bifurcationdetector-1:

BifurcationDetector - 1 (com.microsoft)
=======================================

**Version**

* **name**: `BifurcationDetector (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BifurcationDetector>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Component for aggressive decoding. Find the bifurcation index of predicted tokens, between source tokens,
starting from previous suffix match index, and predicted tokens.
Concat predicted tokens, starting from bifurcation index, to the back
of current tokens. This forms the output tokens.
Detect suffix match index in source tokens, between source tokens and output tokens.
Detection is based on finding the appearances of last n-gram in output tokens
in source tokens.
A match is considered found if source tokens contain a single matching n-gram.
Return the index of the start of the n-gram in source tokens.
No matching if found if src tokens contain multiple or zero matching n-grams. Return -1.

**Attributes**

* **max_ngram_size**:
  The maximum NGram size for suffix matching. Default value is ``?``.
* **min_ngram_size**:
  The minimum NGram size for suffix matching. Default value is ``?``.

**Inputs**

Between 3 and 4 inputs.

* **src_tokens** (heterogeneous) - **T**:
  Encoder input ids.
* **cur_tokens** (heterogeneous) - **T**:
  Decoder input ids.
* **prev_suffix_match_idx** (heterogeneous) - **T**:
  Previous suffix match index
* **pred_tokens** (optional, heterogeneous) - **T**:
  Predicted token ids from aggressive decoding

**Outputs**

* **tokens** (heterogeneous) - **T**:
  Decoder input ids after merging predicted tokens
* **suffix_match_idx** (heterogeneous) - **T**:
  new suffix match index

**Examples**
