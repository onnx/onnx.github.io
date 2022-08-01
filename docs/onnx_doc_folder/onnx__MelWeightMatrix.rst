
.. _l-onnx-doc-MelWeightMatrix:

===============
MelWeightMatrix
===============

.. contents::
    :local:


.. _l-onnx-op-melweightmatrix-17:

MelWeightMatrix - 17
====================

**Version**

* **name**: `MelWeightMatrix (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MelWeightMatrix>`_
* **domain**: **main**
* **since_version**: **17**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 17**.

**Summary**

Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
This function defines the mel scale in terms of a frequency in hertz according to the following formula:

    mel(f) = 2595 * log10(1 + f/700)

In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.

The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of linear scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram" M of shape [frames, num_mel_bins].

**Attributes**

* **output_datatype**:
  The data type of the output tensor. Strictly must be one of the
  values from DataType enum in TensorProto whose values correspond to
  T3. The default value is 1 = FLOAT. Default value is ``1``.

**Inputs**

* **num_mel_bins** (heterogeneous) - **T1**:
  The number of bands in the mel spectrum.
* **dft_length** (heterogeneous) - **T1**:
  The size of the original DFT. The size of the original DFT is used
  to infer the size of the onesided DFT, which is understood to be
  floor(dft_length/2) + 1, i.e. the spectrogram only contains the
  nonredundant DFT bins.
* **sample_rate** (heterogeneous) - **T1**:
  Samples per second of the input signal used to create the
  spectrogram. Used to figure out the frequencies corresponding to
  each spectrogram bin, which dictates how they are mapped into the
  mel scale.
* **lower_edge_hertz** (heterogeneous) - **T2**:
  Lower bound on the frequencies to be included in the mel spectrum.
  This corresponds to the lower edge of the lowest triangular band.
* **upper_edge_hertz** (heterogeneous) - **T2**:
  The desired top edge of the highest frequency band.

**Outputs**

* **output** (heterogeneous) - **T3**:
  The Mel Weight Matrix. The output has the shape:
  [floor(dft_length/2) + 1][num_mel_bins].

**Type Constraints**

* **T1** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain to integer tensors.
* **T2** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain to float tensors
* **T3** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain to any numerical types.

**Examples**
