
.. _l-onnx-doc-DFT:

===
DFT
===

.. contents::
    :local:


.. _l-onnx-op-dft-17:

DFT - 17
========

**Version**

* **name**: `DFT (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#DFT>`_
* **domain**: **main**
* **since_version**: **17**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 17**.

**Summary**

Computes the discrete Fourier transform of input.

**Attributes**

* **axis**:
  The axis on which to perform the DFT. By default this value is set
  to 1, which corresponds to the first dimension after the batch
  index. Default value is ``1``.
* **inverse**:
  Whether to perform the inverse discrete fourier transform. By
  default this value is set to 0, which corresponds to false. Default value is ``0``.
* **onesided**:
  If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2)
  + 1] are returned because the real-to-complex Fourier transform
  satisfies the conjugate symmetry, i.e., X[m, w] =
  X[m,w]=X[m,n_fft-w]*. Note if the input or window tensors are
  complex, then onesided output is not possible. Enabling onesided
  with real inputs performs a Real-valued fast Fourier transform
  (RFFT). When invoked with real or complex valued input, the default
  value is 0. Values can be 0 or 1. Default value is ``0``.

**Inputs**

Between 1 and 2 inputs.

* **input** (heterogeneous) - **T1**:
  For real input, the following shape is expected:
  [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1]. For
  complex input, the following shape is expected:
  [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]. The first
  dimension is the batch dimension. The following N dimentions
  correspond to the signal's dimensions. The final dimension
  represents the real and imaginary parts of the value in that order.
* **dft_length** (optional, heterogeneous) - **T2**:
  The length of the signal.If greater than the axis dimension, the
  signal will be zero-padded up to dft_length. If less than the axis
  dimension, only the first dft_length values will be used as the
  signal. It's an optional value.

**Outputs**

* **output** (heterogeneous) - **T1**:
  The Fourier Transform of the input vector.If onesided is 0, the
  following shape is expected:
  [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]. If axis=0
  and onesided is 1, the following shape is expected:
  [batch_idx][floor(signal_dim1/2)+1][signal_dim2]...[signal_dimN][2].
  If axis=1 and onesided is 1, the following shape is expected:
  [batch_idx][signal_dim1][floor(signal_dim2/2)+1]...[signal_dimN][2].
  If axis=N-1 and onesided is 1, the following shape is expected:
  [batch_idx][signal_dim1][signal_dim2]...[floor(signal_dimN/2)+1][2].
  The signal_dim at the specified axis is equal to the dft_length.

**Type Constraints**

* **T1** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **T2** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain scalar length types to int64_t.

**Examples**
