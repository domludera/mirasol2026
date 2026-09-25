# Third-party notices and attribution

This file satisfies the notice obligations of every upstream whose code, weights or
data reaches this repository. It is a licence document: please do not condense it, and
do not remove the copyright headers it refers to.

`LICENSE` grants MIT over this repository's own code. Everything listed below is
covered by the licence stated in its own section instead.

**Contents**

| # | Upstream | Licence | Reaches this repo as |
|---|---|---|---|
| 1 | [HTS-AT](#1-hts-at) | MIT | seven source files with preserved headers, plus four derived files |
| 2 | [Microsoft Swin-Transformer](#2-microsoft-swin-transformer) | MIT | two source files (via HTS-AT) |
| 3 | [PyTorch Image Models (timm)](#3-pytorch-image-models-timm) | Apache-2.0 | vendored primitives in `models/layers.py`; runtime dependency; three sets of pretrained weights |
| 4 | [PANNs / audioset_tagging_cnn](#4-panns--audioset_tagging_cnn) | MIT | `do_mixup` and `interpolate` |
| 5 | [Apple ml-fastvit](#5-apple-ml-fastvit-weights) | Apple's own terms | every released FastViT ONNX graph |
| 6 | [CirCor DigiScope](#6-circor-digiscope-phonocardiogram-dataset) | ODC-By 1.0 | `results/predictions/`, `data/splits/` (derived); dataset itself **not** redistributed |
| 7 | [Other pretrained weights](#7-other-pretrained-weights) | Apache-2.0 / BSD-3-Clause | released ONNX graphs |
| 8 | [Dependency licences](#8-dependency-licences) | various | declared in `pyproject.toml` |

---

## 1. HTS-AT

**Upstream:** <https://github.com/RetroCirce/HTS-Audio-Transformer>
**Licence:** MIT
**Copyright:** (c) 2022 Knut(Ke) Chen (`knutchen@ucsd.edu`)
**Paper:** K. Chen, X. Du, B. Zhu, Z. Ma, T. Berg-Kirkpatrick, S. Dubnov. *HTS-AT: A
Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection.*
ICASSP 2022, pp. 646-650. <https://doi.org/10.1109/ICASSP43922.2022.9746312>

The training core of this repository is a modification of HTS-AT. Seven files begin with
HTS-AT's original attribution block and **must keep those lines byte-for-byte**:

```
# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
```

followed in each file by its original one-line description, and then by this repository's
added modification notice:

```
# SPDX-License-Identifier: MIT
# Derived from HTS-AT, Copyright (c) 2022 Knut(Ke) Chen - see THIRD_PARTY.md
# Modified 2026 for MIRASOL/MICCAI heart-murmur screening
```

| File | Upstream file | Modifications made here |
|---|---|---|
| `src/mirasol/config/defaults.py` | `config/esc_config.py` | values rewritten for the CirCor 3-class task; dead keys removed; made read-only defaults behind a CLI/YAML override layer |
| `src/mirasol/cli/train.py` | `main_moody.py` | AudioSet/ensemble/weight-averaging subcommands removed; CPU fallback; argparse override layer |
| `src/mirasol/data/dataset.py` | `src/data_generator.py` | AudioSet `SEDDataset` removed; only the ESC-style `.npy` dataset retained |
| `src/mirasol/training/lightning_module.py` | `src/sed_model.py` | `Ensemble_SEDWrapper` and AudioSet/SCv2 metric branches removed; Lightning 2.x API |
| `src/mirasol/training/utils.py` | `src/utils.py` | dead helpers and the `museval` import removed |
| `src/mirasol/models/htsat.py` | `model/htsat.py` | dead branches removed; `numpy.core.fromnumeric` import removed |
| `src/mirasol/models/layers.py` | `model/layers.py` | unchanged in substance (see section 3 for the timm origin of its contents) |

`config/esc_config.py` is the most exposed of these: its *values* are entirely rewritten
for this project, so the file reads like new code while remaining a derivative of an
HTS-AT file. The header stays.

Four further files are derived from HTS-AT without carrying the original header, because
they were written for this project against HTS-AT's interfaces or re-derive a specific
piece of its logic. They are recorded here instead:

| File | Relationship to HTS-AT |
|---|---|
| `src/mirasol/deployment/export.py` | its `HTSATExportWrapper` re-derives `reshape_wav2img` from `model/htsat.py` (also Swin-derived, see section 2) |
| `src/mirasol/models/frontend.py` | the torchlibrosa spectrogram front end plus `bn0` follows HTS-AT's arrangement |
| `src/mirasol/models/timm_models.py` | same front end, adapting timm backbones to HTS-AT's `BaseAudioModel` interface |
| `src/mirasol/models/efficientnet.py` | same front end, adapting a torchvision backbone |

`knutchen@ucsd.edu` in those headers is a **copyright notice**, not a leaked personal
address. It is reproduced under the MIT licence's requirement to retain the copyright
notice, and it must not be scrubbed.

### MIT License (HTS-AT)

```
MIT License

Copyright (c) 2022 Knut(Ke) Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Microsoft Swin-Transformer

**Upstream:** <https://github.com/microsoft/Swin-Transformer>
**Licence:** MIT
**Copyright:** (c) Microsoft Corporation
**Paper:** Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo. *Swin
Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
<https://arxiv.org/abs/2103.14030>

HTS-AT's model core and layer library are themselves based on Swin-Transformer, and both
files carry Swin's notice on lines 5-6 in addition to the HTS-AT block of section 1:

```
# below codes are based and referred from https://github.com/microsoft/Swin-Transformer
# Swin Transformer for Computer Vision: https://arxiv.org/pdf/2103.14030.pdf
```

| File | Swin-derived content |
|---|---|
| `src/mirasol/models/htsat.py` | windowed attention, patch merging, the hierarchical stage structure |
| `src/mirasol/models/layers.py` | `PatchEmbed`, `Mlp`, `DropPath`, the truncated-normal and variance-scaling initialisers |

`src/mirasol/deployment/export.py` re-derives the `reshape_wav2img` spectrogram-to-image
reshaping, which is part of the same lineage.

### MIT License (Swin-Transformer)

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

---

## 3. PyTorch Image Models (timm)

**Upstream:** <https://github.com/huggingface/pytorch-image-models>
**Licence:** Apache License, Version 2.0
**Copyright:** Copyright 2019 Ross Wightman

timm reaches this repository in three distinct ways.

### 3a. Vendored primitives (source redistribution)

`src/mirasol/models/layers.py` contains primitives that originate in timm and arrived
here through HTS-AT and Swin-Transformer, which vendored them the same way:

```
_ntuple, drop_path, DropPath, PatchEmbed, Mlp,
_no_grad_trunc_normal_, trunc_normal_, variance_scaling_, lecun_normal_
```

**NOTICE (Apache-2.0 section 4(d)):**

```
PyTorch Image Models (timm)
Copyright 2019 Ross Wightman

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

**CHANGE STATEMENT (Apache-2.0 section 4(b)):** the primitives in
`src/mirasol/models/layers.py` are modified from their original timm form. The changes
were made upstream in HTS-AT and Swin-Transformer, and are carried forward here:

- `PatchEmbed` accepts a `patch_stride` distinct from `patch_size`, so patches may
  overlap; the original timm implementation assumes `stride == patch_size`.
- `PatchEmbed` accepts non-square inputs and a `padding` argument, needed for the
  (2001, 64) log-mel input of this project rather than a square image.
- Docstrings and type annotations were added in 2026 for this release, and the module was
  moved from `model/layers.py` to `src/mirasol/models/layers.py`.
- No functional change was made to `drop_path`, `Mlp`, `trunc_normal_`,
  `variance_scaling_` or `lecun_normal_`.

### 3b. Runtime dependency

`timm==1.0.24` is a declared dependency of the `[train]` extra. It supplies the FastViT,
TinyViT, MobileNetV3-Small and ResNet-18 backbones at runtime; those files are not
vendored here.

### 3c. Pretrained weights

Three of the six released architectures were fine-tuned from timm's ImageNet weights, so
the corresponding ONNX graphs are derivative works of Apache-2.0 material: see sections 5
and 7.

### Apache License 2.0 (full text)

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "{}"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2019 Ross Wightman

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

---

## 4. PANNs / audioset_tagging_cnn

**Upstream:** <https://github.com/qiuqiangkong/audioset_tagging_cnn>
**Licence:** MIT
**Copyright:** (c) Qiuqiang Kong
**Paper:** Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M.D. Plumbley. *PANNs: Large-Scale
Pretrained Audio Neural Networks for Audio Pattern Recognition.* IEEE/ACM TASLP 28,
2880-2894 (2020). <https://doi.org/10.1109/TASLP.2020.3030497>

Two helper functions in this repository are PANNs code, inherited through HTS-AT:

| Function | Location |
|---|---|
| `do_mixup` | `src/mirasol/training/utils.py`, and a local fallback definition in `src/mirasol/models/timm_models.py` and `src/mirasol/models/efficientnet.py` |
| `interpolate` | `src/mirasol/training/utils.py` |

The `torchlibrosa` dependency (`Spectrogram`, `LogmelFilterBank`, `SpecAugmentation`),
which computes the log-mel front end of every model here, is by the same author
(Copyright (c) 2018-2021 Qiuqiang Kong) and is also MIT-licensed; it is a declared dependency and is not vendored.

No PANNs *model* code is distributed. An earlier state of the source tree contained an
unattributed verbatim copy of PANNs' `models.py`; it was dead code and is not part of
this release.

### The MIT License (PANNs / audioset_tagging_cnn)

```
The MIT License

Copyright (c) 2018-2020 Qiuqiang Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## 5. Apple ml-fastvit weights

**Upstream:** <https://github.com/apple/ml-fastvit>
**Licence:** Apple's own licence, published in that repository. **It is not the MIT
licence and not Apache-2.0.** Consult
<https://github.com/apple/ml-fastvit/blob/main/LICENSE> before redistributing.
**Copyright:** Copyright (C) 2023 Apple Inc. All Rights Reserved.
**Paper:** P.K.A. Vasu, J. Gabriel, J. Zhu, O. Tuzel, A. Ranjan. *FastViT: A Fast Hybrid
Vision Transformer using Structural Reparameterization.* ICCV 2023.

timm's own header records the provenance verbatim:

```
# FastViT for PyTorch
#
# Original implementation and weights from https://github.com/apple/ml-fastvit
#
# For licensing see accompanying LICENSE file at https://github.com/apple/ml-fastvit/tree/main
# Original work is copyright (C) 2023 Apple Inc. All Rights Reserved.
```

**Why this section exists.** The FastViT arm of this study is initialised from
`fastvit_t8.apple_in1k`. Every FastViT artefact released with this work -
`fastvit/model_fp32.onnx`, `model_fp16.onnx`, `model_int8.onnx`,
`model_int8_static.onnx` - is therefore a derivative of Apple's weights. An unqualified
MIT statement over those files would purport to relicense material this project does not
own. It does not: the MIT grant in `LICENSE` covers this repository's code, and the
FastViT graphs remain subject to Apple's terms in addition. Users who redistribute or
build products on the FastViT graphs must satisfy Apple's licence themselves.

**Summary of Apple's terms, not a quotation.** Apple's licence is a bespoke Apple
licence, not one of the standard OSI texts, so it is not reproduced here; reproducing a
paraphrase as though it were the licence text would be worse than pointing at the
original. Read
<https://github.com/apple/ml-fastvit/blob/main/LICENSE> before redistributing any FastViT
artefact. In outline, that licence:

- is offered under Apple's copyrights in the original Apple software, on the condition
  that using, installing, modifying or redistributing the software constitutes acceptance
  of its terms;
- grants a personal, non-exclusive licence to use, reproduce, modify and redistribute in
  source and/or binary form, with or without modifications;
- requires that redistribution of the software in unmodified and entire form retain the
  copyright notice, the licence text and the disclaimers;
- forbids using Apple's name, trademarks, service marks or logos to endorse or promote
  products derived from the software without prior written permission;
- grants no other rights by implication, and expressly no patent rights;
- disclaims all warranties, express and implied, including non-infringement,
  merchantability and fitness for a particular purpose, and provides the software on an
  "AS IS" basis;
- excludes liability for special, indirect, incidental or consequential damages; and
- notes that ml-fastvit itself bundles subcomponents under separate notices, listed in
  that repository's `ACKNOWLEDGEMENTS` file.

Apple's file governs. If anything in this summary conflicts with it, Apple's file wins.

---

## 6. CirCor DigiScope Phonocardiogram Dataset

**Upstream:** <https://physionet.org/content/circor-heart-sound/>
**Licence:** Open Data Commons Attribution License v1.0 (ODC-By 1.0),
<https://opendatacommons.org/licenses/by/1-0/>

**The dataset is not redistributed here.** No CirCor audio file and no CirCor metadata
file (`training_data.csv`, `train_meta_data.csv`, `RECORDS`, `SHA256SUMS.txt`) is
committed to this repository. `.gitignore` blocks `data/**` for exactly this reason.
Obtain the dataset from PhysioNet under its own terms; see `data/README.md`.

### Derived data that IS committed (ODC-By 1.0, not MIT)

| Path | What it derives from CirCor | Rows |
|---|---|---|
| `results/predictions/*.data.csv` | recording filename (patient identifier + auscultation location) and the murmur target derived from CirCor's `Murmur` column, alongside this project's own logits | 61 files |
| `data/splits/folds.csv` | the CirCor patient identifiers, partitioned into five folds | 941 |

These are Derivative Databases under ODC-By 1.0. They ship because they are what makes
every table in the paper reproducible without downloading the dataset; they add no information
that is not already public in CirCor. If you redistribute them, or a database derived
from them, ODC-By 1.0 applies and the attribution below must travel with them. Each
directory carries a `README.md` restating this, and `LICENSE` section 3 records the same
scope split.

### Required attribution

Cite all three of the following, as PhysioNet's terms for this dataset require:

> Oliveira, J., Renna, F., Costa, P.D., Nogueira, M., Oliveira, C., Ferreira, C.,
> Jorge, A., Mattos, S., Hatem, T., Tavares, T., Elola, A., Rad, A.B., Sameni, R.,
> Clifford, G.D., Coimbra, M.T. The CirCor DigiScope dataset: from murmur detection to
> murmur classification. *IEEE Journal of Biomedical and Health Informatics* 26(6),
> 2524-2535 (2022). <https://doi.org/10.1109/JBHI.2021.3137048>

> Reyna, M.A., Kiarashi, Y., Elola, A., Oliveira, J., Renna, F., Gu, A., Perez Alday,
> E.A., Sadr, N., Sharma, A., Kpodonu, J., Mattos, S., Coimbra, M.T., Sameni, R.,
> Rad, A.B., Clifford, G.D. Heart murmur detection from phonocardiogram recordings: the
> George B. Moody PhysioNet Challenge 2022. *PLOS Digital Health* 2(9), e0000324 (2023).
> <https://doi.org/10.1371/journal.pdig.0000324>

> Goldberger, A.L., Amaral, L.A.N., Glass, L., Hausdorff, J.M., Ivanov, P.C., Mark, R.G.,
> Mietus, J.E., Moody, G.B., Peng, C.-K., Stanley, H.E. PhysioBank, PhysioToolkit, and
> PhysioNet: components of a new research resource for complex physiologic signals.
> *Circulation* 101(23), e215-e220 (2000).
> <https://doi.org/10.1161/01.CIR.101.23.e215>

The screening campaigns that produced the data were approved by the Complexo Hospitalar
HUOC/PROCAPE institutional review board (Pernambuco, Brazil). No new human-subject data
were collected for this study.

---

## 7. Other pretrained weights

Four architectures were fine-tuned from third-party ImageNet weights. The released ONNX
graphs are derivative works of those weights, and the terms below apply to them in
addition to this repository's MIT grant.

| Architecture | Pretrained tag | Source | Licence of the weights |
|---|---|---|---|
| FastViT | `fastvit_t8.apple_in1k` | Apple ml-fastvit via timm | Apple's own terms - see section 5 |
| TinyViT | `tiny_vit_5m_224` | timm, adapted from microsoft/Cream | Apache-2.0 (timm distribution) |
| MobileNetV3-Small | `mobilenetv3_small_100.lamb_in1k` | timm | Apache-2.0 |
| ResNet-18 | `resnet18.a1_in1k` | timm | Apache-2.0 |
| EfficientNet-B0 | `EfficientNet_B0_Weights.IMAGENET1K_V1` | torchvision | BSD-3-Clause |
| HTS-AT | none - trained from scratch | - | architecture is MIT, section 1 |

`tiny_vit_5m_224` is timm's port of TinyViT from
<https://github.com/microsoft/Cream/tree/main/TinyViT> (Microsoft, MIT); the weights are
redistributed by timm under Apache-2.0.

---

## 8. Dependency licences

Declared in `pyproject.toml` and pinned to the versions the published results were
produced with. Each is installed from PyPI under its own licence. None is vendored
wholesale, but portions of **librosa** are ported into this repository - see section 9. Licences were read from the installed distribution metadata of the verified
environment.

| Package | Version | Licence |
|---|---|---|
| numpy | 1.26.4 | BSD-3-Clause |
| scipy | 1.17.0 | BSD-3-Clause |
| pandas | 3.0.1 | BSD-3-Clause |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| matplotlib | 3.10.8 | Matplotlib licence (PSF-based, BSD-compatible) |
| torch | 2.10.0 | BSD-3-Clause |
| torchvision | 0.25.0 | BSD-3-Clause |
| timm | 1.0.24 | Apache-2.0 (see section 3) |
| pytorch-lightning | 2.6.1 | Apache-2.0 |
| torchlibrosa | 0.1.0 | MIT (Qiuqiang Kong, see section 4) |
| librosa | 0.11.0 | ISC |
| soundfile | 0.13.1 | BSD-3-Clause |
| tqdm | 4.67.3 | MPL-2.0 AND MIT |
| tensorboard | 2.19.0 | Apache-2.0 |
| onnx | 1.20.1 | Apache-2.0 |
| onnxruntime | 1.24.1 (1.24.2 on device) | MIT |
| onnxconverter-common | 1.16.0 | MIT |
| ruff, pytest (dev only) | see `pyproject.toml` | MIT |

---

## Reporting an attribution problem

If you believe an upstream is missing from this file, or that a notice here is incomplete
or incorrect, please open an issue at
<https://github.com/domludera/mirasol2026/issues>. Attribution defects will be fixed
ahead of anything else in the backlog.

---

## 9. librosa (ISC) - ported source

`src/mirasol/signal/mel.py` and `deployment/edge/mel.py` port librosa's short-time Fourier
transform and Slaney mel filterbank into NumPy so that the preprocessing chain runs with no
librosa dependency. This matters for `deployment/edge/`, which is copied to the Orange Pi on
its own and must install nothing beyond ONNX Runtime, NumPy, SciPy and soundfile.

The ported functions reproduce librosa's behaviour deliberately and exactly - including its
centred-framing conventions and its `MAX_MEM_BLOCK` chunking - because every published number
depends on the input representation being bit-identical to the one used at training time.

Upstream: <https://github.com/librosa/librosa>

```
Copyright (c) 2013--2023, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any purpose with or
without fee is hereby granted, provided that the above copyright notice and this permission
notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
OR PERFORMANCE OF THIS SOFTWARE.
```
