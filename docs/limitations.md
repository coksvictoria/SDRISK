# Limitations

Table-level identity and attribute metrics (M3–M4), subset analysis, seeded
cluster analysis, internal row profiles (M5), and membership and simplified
attack diagnostics (M6) are implemented. The no-holdout ML attribute check is
experimental and is not a synthetic-data leakage estimate. Cluster exposure
ratio (CER), KDE density-ratio MIA, singling-out, and linkability are
experimental diagnostics. CER has not been validated on a dedicated
localized-leak benchmark. The simplified Anonymeter-style attacks do not
include randomized predicate generation, auxiliary tables, or naive attacks.
Distance and density membership metrics require a holdout. Row-band thresholds
must be set in a user-supplied profile and confirmed as policy.

Decision classifications use the configured metric thresholds. Metrics without
a configured threshold remain in review; the package does not invent a default
pass/fail cutoff. Recommendations and confidence are operational heuristics,
not approvals or calibrated statistical guarantees. Confidence is reduced by
missing holdout data, unassessed or suppressed results, and unstable cluster
fits. Public-tier recommendations are capped when the profile requires a
holdout and none was supplied.

The CLI recomputes assessments during remediation. Its run manifest records
source paths to support that operation and must remain in the restricted
environment. JSON and HTML reports contain filtered aggregates, while row-level
profiles remain in memory. Cleaning returns only the filtered synthetic
dataset; it does not export row identifiers, flags, or counts of removed rows.

Passing these checks lowers concern but does not prove the data cannot be disclosive.
Thresholds are policy choices. Attacks requiring generator access were not run.
