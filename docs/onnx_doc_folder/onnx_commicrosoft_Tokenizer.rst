
.. _l-onnx-doccom.microsoft-Tokenizer:

=========================
com.microsoft - Tokenizer
=========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-tokenizer-1:

Tokenizer - 1 (com.microsoft)
=============================

**Version**

* **name**: `Tokenizer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Tokenizer>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

  Tokenizer divides each string in X into a vector of strings along the last axis. Allowed input shapes are [C] and [N, C].
  If the maximum number of tokens found per input string is D, the output shape would be [N, C, D] when input shape is [N, C].
  Similarly, if input shape is [C] then the output should be [C, D]. Tokenizer has two different operation modes.
  The first mode is selected when "tokenexp" is not set and "separators" is set. If "tokenexp" is set and "separators" is not set,
  the second mode will be used. The first mode breaks each input string into tokens by matching and removing separators.
  "separators" is a list of strings which are regular expressions. "tokenexp" is a single regular expression.
  Let's assume "separators" is [" "] and consider an example.
  If input is
  ["Hello World", "I love computer science !"] whose shape is [2],
  then the output would be
 [["Hello", "World", padvalue, padvalue, padvalue],
 ["I", "love", "computer", "science", "!"]]
 whose shape is [2, 5] because you can find at most 5 tokens per input string.
 Note that the input at most can have two axes, so 3-D and higher dimension are not supported.
 If "separators" contains a single empty string, the Tokenizer will enter into character tokenezation mode. This means all strings
 will be broken part into individual characters.
 For each input string, the second mode searches matches of "tokenexp" and each match will be a token in Y.
 The matching of "tokenexp" is conducted greedily (i.e., a match should be as long as possible).
 This operator searches for the first match starting from the beginning of the considered string,
 and then launches another search starting from the first remained character after the first matched token.
 If no match found, this operator will remove the first character from the remained string and do another search.
 This procedure will be repeated until reaching the end of the considered string.
  Let's consider another example to illustrate the effect of setting "mark" to true.
  If input is ["Hello", "World"],
  then the corresponding output would be [0x02, "Hello", "World", 0x03].
  This implies that if mark is true, [C]/[N, C] - input's output shape becomes [C, D+2]/[N, C, D+2].
If tokenizer removes the entire content of [C]-input, it will produce [[]].
I.e. the output shape should be [C][0] or [N][C][0] if input shape was [N][C].
If the tokenizer receives empty input of [0] then the output is [0] if empty input
of [N, 0] then [N, 0].

**Attributes**

* **mark** (required):
  Boolean whether to mark the beginning/end character with start of
  text character (0x02)/end of text character (0x03). Default value is ``?``.
* **mincharnum** (required):
  Minimum number of characters allowed in the output. For example, if
  mincharnum is 2, tokens such as "A" and "B" would be ignored Default value is ``?``.
* **pad_value** (required):
  The string used to pad output tensors when the tokens extracted
  doesn't match the maximum number of tokens found. If start/end
  markers are needed, padding will appear outside the markers. Default value is ``?``.
* **separators**:
  an optional list of strings attribute that contains a list of
  separators - regular expressions to match separators Two consecutive
  segments in X connected by a separator would be divided into two
  tokens. For example, if the input is "Hello World!" and this
  attribute contains only one space character, the corresponding
  output would be ["Hello", "World!"]. To achieve character-level
  tokenization, one should set the 'separators' to [""], which
  contains an empty string. Default value is ``?``.
* **tokenexp**:
  An optional string. Token's regular expression in basic POSIX format
  (pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html#ta
  g_09_03). If set, tokenizer may produce tokens matching the
  specified pattern. Note that one and only of 'tokenexp' and
  'separators' should be set. Default value is ``?``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Strings to tokenize

**Outputs**

* **Y** (heterogeneous) - **T**:
  Tokenized strings

**Examples**
