# Accelerated Part 0 human validation

This private, offline workflow connects the current accelerated Part 0 JSONL
evidence to two genuine human annotators and one independent adjudicator. It
does not call an LLM, invent labels, modify the source run, or convert the
exploratory campaign into a confirmatory experiment.

Packets contain harmful requests and private model responses. Keep the output
on nonsynchronized local storage; do not commit, upload, attach, or release it.

## 1. Export two blinded packets

```bash
uv run python -m analysis.accelerated_part0_human_validation export \
  --run-dir data/private/inference_hub/part0-sota-panel-v2-n24 \
  --output-dir /path/to/private/accelerated-part0-human-audit
```

The adapter revalidates the source manifest, schedule hash, every JSONL hash
chain and checkpoint, subject/response identity, fixed-judge route and batch,
and visible prompt/response hashes. Operationally retired subjects are excluded
as whole subjects. Every valid fixed-judge row from every remaining
subject-by-language stratum is included: this is a census, not an
outcome-selected sample.

The source campaign crossed archived English requests with requested-response
language instructions; it did not retain translated requests. Accordingly,
`request_original` and `request_english` are the same English request, while a
non-English `response_original` has no machine-filled English translation. A
language-qualified reviewer must judge the original response itself.

It creates:

- `annotator_1_packet.csv` and `annotator_2_packet.csv`, independently ordered
  with blank human-label fields;
- one delayed duplicate packet per annotator, sampled proportionally across
  subject-by-language strata;
- `audit_key.csv`, which must remain hidden from annotators;
- `language_qualifications.template.json`; and
- a private manifest that records the source hashes and stratum counts.

The annotation packets contain no subject/provider/model identity, fixed-judge
identity or label, source-row index, or duplicate-to-primary link. Primary and
delayed item IDs are independently generated, cryptographically random opaque
identifiers—not hashes that an annotator can recompute from the public roster.
Both annotators independently fill every `annotation_label`, `confidence`, and
`error_type` field under the definitions in [JUDGE_AUDIT.md](JUDGE_AUDIT.md).
They complete their delayed packets later without access to their first pass.

## 2. Record genuine language qualifications

Copy `language_qualifications.template.json` to a new private
`language_qualifications.json` and complete it from real qualification records.
Both annotators and the independent adjudicator must explicitly attest that
they are human and qualified in English, Chinese, and Russian. Annotator IDs
must remain `annotator_1` and `annotator_2`; the adjudicator must be a different
person. Placeholder or incomplete attestations fail closed.

The file is an auditable assertion, not proof of identity or fluency. Retain
the underlying qualification evidence under the approved research-data policy.

## 3. Prepare disagreement-only adjudication

After both primary packets are completed and locked:

```bash
uv run python -m analysis.accelerated_part0_human_validation prepare-adjudication \
  --run-dir data/private/inference_hub/part0-sota-panel-v2-n24 \
  --key /path/to/private/accelerated-part0-human-audit/audit_key.csv \
  --annotator annotator_1=/path/to/private/annotator_1.completed.csv \
  --annotator annotator_2=/path/to/private/annotator_2.completed.csv \
  --output /path/to/private/adjudication.csv
```

The adjudicator completes every disagreement. The packet remains blind to
subject identity and the fixed automated label. A header-only file is valid
when the annotators have no disagreements.

## 4. Score only complete qualified annotations

```bash
uv run python -m analysis.accelerated_part0_human_validation score \
  --run-dir data/private/inference_hub/part0-sota-panel-v2-n24 \
  --key /path/to/private/accelerated-part0-human-audit/audit_key.csv \
  --annotator annotator_1=/path/to/private/annotator_1.completed.csv \
  --annotator annotator_2=/path/to/private/annotator_2.completed.csv \
  --duplicate-annotations annotator_1=/path/to/private/annotator_1_duplicates.completed.csv \
  --duplicate-annotations annotator_2=/path/to/private/annotator_2_duplicates.completed.csv \
  --adjudications /path/to/private/adjudication.completed.csv \
  --qualifications /path/to/private/language_qualifications.json \
  --output /path/to/private/accelerated_part0_human_validation.json
```

No output is written if a required label, confidence, error type, adjudication,
source binding, or language qualification is missing or altered. A successful
report contains:

- human-versus-fixed-judge confusion matrices and accuracy overall and by
  language;
- binary and three-class precision, recall, balanced accuracy, and macro-F1;
- inter-rater and delayed-duplicate intra-rater agreement overall and by
  language;
- bootstrap intervals, human/automated unclear rates, source/input hashes; and
- `human_validation_complete: true` plus the passed language-qualification
  gate.

The report intentionally retains
`confirmatory_or_paper_promotion_permitted: false`. It validates the fixed judge
on this accelerated exploratory population but cannot repair the absent benign
controls, response-language-only design, or other source-campaign limitations.
