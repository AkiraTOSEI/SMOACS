# THIRD_PARTY_NOTICES.md

SMOACS original code is distributed under the MIT License in `LICENSE`.
Third-party files included in this repository are governed by their respective
licenses or terms, as described below. Pretrained model weights are not
vendored in this repository.

## CrystalFormer-Derived Code

- Path: `src/crystalformer/`
- Upstream URL: https://github.com/deepmodeling/CrystalFormer
- License: Apache-2.0 license
- Local use: SMOACS uses CrystalFormer-derived model/runtime code for
  CrystalFormer predictor support.
- Local changes: integrated into the SMOACS package layout and call sites.

The CrystalFormer-derived files are not covered by the SMOACS MIT License.
They remain subject to the upstream Apache-2.0 license.

## ALIGNN Minimal Runtime Code

- Path: `third_party/alignn_minimal/`
- Upstream URL: https://github.com/usnistgov/alignn
- License/terms: see the upstream ALIGNN license:
  https://github.com/usnistgov/alignn/blob/main/LICENSE.rst
- Local use: SMOACS uses a minimal vendored runtime subset for ALIGNN predictor
  support without requiring the external `alignn` Python package at runtime.

Vendored files:

- `third_party/alignn_minimal/LICENSE.rst`
- `third_party/alignn_minimal/graphs.py`
- `third_party/alignn_minimal/models/alignn.py`
- `third_party/alignn_minimal/models/utils.py`
- `third_party/__init__.py`
- `third_party/alignn_minimal/__init__.py`
- `third_party/alignn_minimal/models/__init__.py`

Local changes:

- Internal imports were adjusted from `alignn.*` to
  `third_party.alignn_minimal.*`.
- SMOACS import sites were adjusted to import the vendored minimal runtime.
- ALIGNN files outside the runtime subset needed by SMOACS are not vendored.

The ALIGNN-derived files are not covered by the SMOACS MIT License. They remain
subject to the upstream ALIGNN license/terms linked above and to the notice in
`third_party/alignn_minimal/LICENSE.rst`.

## External Dependencies Not Vendored

### ALIGNN Pretrained Weights

ALIGNN pretrained weights are not committed to this repository. Download URLs
and expected local paths are listed in `manifests/model_weights_manifest.tsv`.
Those weights are governed by their upstream terms.

### MatterSim

- Upstream URL: https://github.com/microsoft/mattersim
- Usage: optional energy and relaxation backend.
- Vendored code: not included in this repository.
- License: MIT

The differentiable MatterSim implementation used by the SMOACS energy loss is
installed as an external dependency or local checkout. MatterSim checkpoints are
not committed to this repository.

## Excluded Third-Party Comparison Code

CDVAE, FTCP, and MatterGen code are not included in this GitHub repository.
