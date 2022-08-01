
.. _l-onnx-doccom.microsoft-BeamSearch:

==========================
com.microsoft - BeamSearch
==========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-beamsearch-1:

BeamSearch - 1 (com.microsoft)
==============================

**Version**

* **name**: `BeamSearch (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BeamSearch>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Beam Search for text generation. Supports GPT-2 decoder.

**Attributes**

* **decoder** (required):
  Decoder subgraph to execute in a loop. Default value is ``?``.
* **early_stopping**:
  early stop or not Default value is ``?``.
* **encoder_decoder_init**:
  subgraph for initialization of encoder and decoder. It will be
  called once before decoder subgraph. Default value is ``?``.
* **eos_token_id** (required):
  The id of the end-of-sequence token Default value is ``?``.
* **model_type**:
  model type: 0 for GPT-2; 1 for encoder decoder like T5 Default value is ``?``.
* **no_repeat_ngram_size**:
  no repeat ngrams size Default value is ``?``.
* **pad_token_id** (required):
  The id of the padding token Default value is ``?``.

**Inputs**

Between 6 and 10 inputs.

* **input_ids** (heterogeneous) - **I**:
  The sequence used as a prompt for the generation. Shape is
  (batch_size, sequence_length)
* **max_length** (heterogeneous) - **I**:
  The maximum length of the sequence to be generated. Shape is (1)
* **min_length** (optional, heterogeneous) - **I**:
  The minimum length below which the score of eos_token_id is set to
  -Inf. Shape is (1)
* **num_beams** (heterogeneous) - **I**:
  Number of beams for beam search. 1 means no beam search. Shape is
  (1)
* **num_return_sequences** (heterogeneous) - **I**:
  The number of returned sequences in the batch. Shape is (1)
* **temperature** (heterogeneous) - **T**:
  The value used to module the next token probabilities. Accepts value
  > 0.0. Shape is (1)
* **length_penalty** (optional, heterogeneous) - **T**:
  Exponential penalty to the length. Default value 1.0 means no
  penalty.Value > 1.0 encourages longer sequences, while values < 1.0
  produces shorter sequences.Shape is (1,)
* **repetition_penalty** (optional, heterogeneous) - **T**:
  The parameter for repetition penalty. Default value 1.0 means no
  penalty. Accepts value > 0.0. Shape is (1)
* **vocab_mask** (optional, heterogeneous) - **M**:
  Mask of vocabulary. Words that masked with 0 are not allowed to be
  generated, and 1 is allowed. Shape is (vacab_size)
* **prefix_vocab_mask** (optional, heterogeneous) - **M**:
  Mask of vocabulary for first step. Words that masked with 0 are not
  allowed to be generated, and 1 is allowed. Shape is (batch_size,
  vocab_size)

**Outputs**

Between 1 and 3 outputs.

* **sequences** (heterogeneous) - **I**:
  Word IDs of generated sequences. Shape is (batch_size,
  num_return_sequences, max_sequence_length)
* **sequences_scores** (optional, heterogeneous) - **T**:
  Final beam score of the generated sequences. Shape is (batch_size,
  num_return_sequences)
* **scores** (optional, heterogeneous) - **T**:
  Processed beam scores for each vocabulary token at each generation
  step.Beam scores consisting of log softmax scores for each
  vocabulary token and sum of log softmax of previously generated
  tokens in this beam.Shape is (max_length - sequence_length,
  batch_size, num_beams, vocab_size)

**Examples**
