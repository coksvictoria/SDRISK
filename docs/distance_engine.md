# Preprocessing and distance engine

M2 provides internal building blocks for later risk metrics. These functions do
not themselves make a release recommendation or export row-level results.

## Mixed-type encoding

`MixedTypeEncoder` in `src/sdrisk/prep/encoder.py` is fit on reference/training
rows and then used to transform other frames:

- Numeric values are scaled using the fitted minimum and maximum. Constant
  training columns map to zero. Missing values use the fitted median plus a
  separate missingness indicator.
- Dates are interpreted in UTC and represented as days from the Unix epoch
  before scaling.
- Numeric ordinal values use their numeric order. Text ordinal values require a
  pandas categorical column with `ordered=True`; the declared category order is
  used.
- Categorical values use one-hot columns based on levels observed during fit
  (in first-seen training order), plus dedicated missing and unseen-level
  columns.
- Identifier and ignored variables cannot be selected as features.

The encoder does not clip transformed values that fall outside the training
range. Callers should fit it only on the intended reference population.

## Gower dissimilarity

`GowerDistance` in `src/sdrisk/distance/gower.py` uses the schema weights and
columns selected explicitly, or all variables marked `qi` by default. Call
`fit(reference)` before `pairwise(left, right)`.

Numeric, ordinal, and date differences are divided by their non-missing range
in the reference data and capped at one. A constant numeric reference column
uses a binary equality comparison. Categorical dissimilarity is zero for
agreement and one for disagreement. Each pair is the weighted mean across
variables observed in both rows; unavailable variables are removed from that
pair's denominator. A pair with no comparable variables has distance `NaN`.

## Nearest neighbors, blocking, and subsampling

`nearest_neighbor_distances` returns only minimum distances, never matching
record indices. Its query chunk size bounds the pairwise matrix memory, while
the work remains proportional to query rows times reference rows times feature
count. Exact-value blocking limits candidates to matching block values. If a
query has no matching block, it falls back to the full reference by default;
set `fallback_to_global=False` to return `NaN` instead. When a matching block
exists, the result is the nearest distance **within that block**, not a global
nearest-neighbor guarantee.

`subsample_rows` requires an explicit `numpy.random.Generator`, samples without
replacement, restores source row order, and returns the sample fraction with
the sampled frame.

## Generated-data scaling check

The following small timing check was run on Windows x64 with Python 3.11.15,
NumPy 2.4.6, and pandas 3.0.6. It generated a two-column reference dataset
(numeric and categorical), fit ranges before timing, and searched for one query
row. Peak memory is the incremental Python/NumPy allocation reported by
`tracemalloc` during the search; the already-resident reference dataframe is
excluded.

| Reference rows | Query rows | Search time | Incremental peak |
|---:|---:|---:|---:|
| 100,000 | 1 | 0.0100 s | 9.10 MiB |
| 1,000,000 | 1 | 0.0881 s | 102.84 MiB |

These figures are a generated-data sanity check, not a performance guarantee:
they cover one query row and do not estimate a full synthetic-by-training
comparison. Chunking bounds memory, but it does not remove the quadratic
comparison work for similarly sized query and reference sets. Use configured
subsampling for workloads too large for exact comparison. No approximate
nearest-neighbor backend is enabled in M2.
