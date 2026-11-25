# Original logos and other media

This directory includes original logo submissions for use on the ONNX website,
in presentations, and in print. It also contains other media from earlier
versions of the website. A file can be useful even when no current page links
to it.

## Original files and website images

The [member company logo submission instructions](https://github.com/onnx/onnx/blob/main/community/logo_request.md)
ask contributors to place a high-quality logo with a transparent background in
`assets/`: a 300 dpi image suitable for printing or a vector file. They also
require written permission in the PR for website and presentation use.

Images used by current pages are generally under [`images/`](../images/), with
company and tool logos under [`images/logos/`](../images/logos/). These may be
resized or converted versions of the originals. Their smaller dimensions or
different format do not make them equivalent replacements for the source files.

The [2020 website refresh](https://github.com/onnx/onnx.github.io/commit/3fe3c286b24efd02421d76288853e9d60f2f12eb)
introduced the `images/` layout and explicitly restored some original logo files.
The two directories therefore have overlapping content by design.

## Adding or updating a logo

Follow the linked submission instructions and retain the high-quality original
in `assets/`. If the website needs a resized or converted copy, place it under
`images/logos/` and update the relevant page or partial. Describe the relationship
between the original and the website copy in the PR, especially if their names
differ.

## Reviewing cleanup candidates

- Check page, stylesheet, script, and data references, as well as Git history.
  The absence of a website reference does not establish that an original logo
  is obsolete.
- Compare dimensions, format, transparency, and content before treating similar
  filenames as duplicates. Preserve the original when removing a redundant
  resized copy.
- For byte-identical copies, check the documented submission location and any
  existing links before choosing which path to retain. Update affected references
  as part of the same change.
- Review videos, illustrations, and event materials separately from logo
  originals. Repository searches cannot establish whether external sites still
  link directly to their public URLs.
