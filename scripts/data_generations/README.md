# data_generations

This directory contains scripts for generating Materials Project-based initial structures
restricted to elements {B, C, N, O, Si, H, Al} without Wyckoff coordinates.

## Recommended workflow

1) Build oxidation mask (once)
```bash
poetry run python scripts/data_generations/mask_restrict.py
```

2) Prepare a density reference JSON

`density_reference.py` is not part of the current minimal public migration because
the legacy copy needs cleanup before publication. Provide a checked density
reference JSON via the data manifest or regenerate it with a cleaned script.

3) Generate initial structures (cache-first)
```bash
poetry run python scripts/data_generations/generate_initials_superhard.py \
  --n 16 \
  --out_dir data/raw_data \
  --density_ref scripts/data_generations/density_ref_superhard.json \
  --mask_path data/intermidiate/mask_superhard_materials.npz \
  --cache_only true
```

To allow fresh MP fetches when cache is insufficient, set `--cache_only false`
and provide `MP_API_KEY` via environment.
