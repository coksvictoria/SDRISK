# Checker guide

# Checker guide

`sdrisk` provides evidence for a release checker; it does not make the release
decision. Read the result in the context of the approved policy profile,
population, intended release, and plausible attacker knowledge.

## Review the report

1. **Check provenance.** Confirm the profile, seed, configuration hash, and
   input hashes in `audit.json`. The operational `run_manifest.json` contains
   local input paths and must remain restricted.
2. **Check applicability.** A `not_assessed` result means an input or
   precondition was unavailable. A `suppressed` result is intentionally hidden
   by the output policy. Neither is evidence of low risk.
3. **Read each value with its baseline.** Compare table, subset, and cluster
   findings only with their stated baseline. The HTML and JSON reports include
   a plain-language explanation and a suggested submitter question for each
   diagnostic.
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
