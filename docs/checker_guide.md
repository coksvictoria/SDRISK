# Checker guide

# Checker guide

`sdrisk` provides evidence for a release checker; it does not make the release
decision. Read the result in the context of the approved policy profile,
population, intended release, and plausible attacker knowledge.

## Review the report

1. **Check provenance.** Confirm the profile, seed, configuration hash, and
   input hashes in `audit.json`. The operational `run_manifest.json` contains
   local input paths and must remain restricted.
2. **Check applicability.** The report summarizes suppressed and
   `not_assessed` result counts by level. Each affected metric summary retains
   its status and reason; neither status is evidence of low risk.
3. **Read each value with its baseline.** Compare table and cluster findings
   with their stated baselines. Subset results are unweighted medians across
   reportable groups, shown beside output-filtered status counts; they are not
   pooled estimates. QI keys are withheld, while minimum-n suppressions remain
   summarized by the generic QI-set or stratum label. Consult
   [metric reference](./metric_reference.md) for definitions and weaknesses.
4. **Treat experimental results cautiously.** CER, KDE density-ratio MIA,
   singling-out, and linkability are marked experimental. The simplified
   attack variants are not complete implementations of Anonymeter.
5. **Review confidence and recommendation.** A `Release` recommendation means
   all assessed metrics were below configured review thresholds; it is not a
   finding that the data are safe. Missing holdout data, suppressed or
   unassessed results, and unstable cluster fits reduce confidence. For a
   public-tier profile requiring a holdout, the recommendation is capped when
   none is supplied.
6. **Record the checker decision separately.** The checker remains responsible
   for the final decision, rationale, any modifications required, and who
   approved release.

## Remediation

`sdrisk clean` reruns the analysis from the inputs recorded in the run manifest,
checks that source hashes and configuration still match, and writes a cleaned
synthetic file. `--max-band medium` retains low- and medium-band rows and
removes high-band rows. The command does not print or write a removed-row list
or count. Keep the cleaned dataset under the same controls as the input.

## Limitations

Thresholds and output controls are policy choices and must be confirmed for the
target environment. Passing these checks lowers concern but does not prove the
data cannot be disclosive. Attacks requiring generator access were not run.
See [limitations](./limitations.md) for metric and validation limitations.
