# Memory CI/CD Quarantine Case Study

This qualitative case study shows TruthGit treating memory writes like code going through CI.

Run it:

```bash
python -m experiments.memory_ci_case_study --output experiments/results/memory_ci_case_study.json
```

## Scenario

1. A good low-risk update arrives:
   `Alice's public profile says her favorite color is green.`

   Memory CI classifies `favorite_color` as `low_risk`. The source is trusted enough, there is no contradiction, no temporal overlap, no duplicate provenance, and no rollback regression. The staged write passes and is applied.

2. A trusted residence seed arrives:
   `Verified city registry says Alice lives in Busan.`

   Memory CI classifies `lives_in` as `identity_state`, so it requires review on `main`. A reviewer approves it. The resulting belief version has one active support source: `verified-city-registry`.

3. A suspicious update arrives:
   `Anonymous forum joke says Alice lives in Atlantis.`

   Memory CI quarantines it. The source trust is below the fail threshold and the claim conflicts with the stronger active Busan memory. The proposed write is stored durably as a staged commit, but it does not create an active belief version.

4. The reviewer rejects the quarantined write.

   The staged commit becomes `rejected`. The audit trail records creation, Memory CI start/completion, quarantine, and rejection. The Busan belief remains active and keeps its support source.

## What To Show

Open `experiments/results/memory_ci_case_study.json` and point to:

- `steps[0].checks.run.decision = auto_apply`
- `steps[2].checks.run.decision = quarantine`
- `steps[2].status = rejected`
- `belief_versions`: no Atlantis active version exists
- `audit_events`: every transition is recorded with `entity_key` equal to the staged commit UUID

This demonstrates the paper claim: TruthGit is governed truth maintenance, not just retrieval plus post-hoc explanations.
