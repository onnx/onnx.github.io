
.. _l-onnx-doc-STFT:

====
STFT
====

.. contents::
    :local:


.. _l-onnx-op-stft-17:

STFT - 17
=========

**Version**

* **name**: `STFT (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#STFT>`_
* **domain**: **main**
* **since_version**: **17**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 17**.

**Summary**

Computes the Short-time Fourier Transform of the signal.

**Attributes**

* **onesided**:
  If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2)
  + 1] are returned because the real-to-complex Fourier transform
  satisfies the conjugate symmetry, i.e., X[m, w] =
  X[m,w]=X[m,n_fft-w]*. Note if the input or window tensors are
  complex, then onesided output is not possible. Enabling onesided
  with real inputs performs a Real-valued fast Fourier transform
  (RFFT).When invoked with real or complex valued input, the default
  value is 1. Values can be 0 or 1. Default value is ``1``.

**Inputs**

Between 2 and 4 inputs.

* **signal** (heterogeneous) - **T1**:
  Input tensor representing a real or complex valued signal. For real
  input, the following shape is expected:
  [batch_size][signal_length][1]. For complex input, the following
  shape is expected: [batch_size][signal_length][2], where
  [batch_size][signal_length][0] represents the real component and
  [batch_size][signal_length][1] represents the imaginary component of
  the signal.
* **frame_step** (heterogeneous) - **T2**:
  The number of samples to step between successive DFTs.
* **window** (optional, heterogeneous) - **T1**:
  A tensor representing the window that will be slid over the
  signal.The window must have rank 1 with shape: [window_shape]. It's
  an optional value.
* **frame_length** (optional, heterogeneous) - **T2**:
  A scalar representing the size of the DFT. It's an optional value.

**Outputs**

* **output** (heterogeneous) - **T1**:
  The Short-time Fourier Transform of the signals.If onesided is 1,
  the output has the shape: [batch_size][frames][dft_unique_bins][2],
  where dft_unique_bins is frame_length // 2 + 1 (the unique
  components of the DFT) If onesided is 0, the output has the shape:
  [batch_size][frames][frame_length][2], where frame_length is the
  length of the DFT.

**Type Constraints**

* **T1** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain signal and output to float tensors.
* **T2** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain scalar length types to int64_t.

**Examples**
