# How To Update CoreStep Pipeline Behavior

Core step orchestration lives in
`tradingchassis_core/core/domain/processing_step.py`.

Recommended workflow:

1. Start from `run_core_step` and identify which phase changes:
   reduction, Strategy evaluation, reconciliation, Risk Engine (policy), or apply.
2. Keep stage boundaries explicit:
   - reduction first
   - Strategy generation second
   - candidate reconciliation third
   - Risk Engine (policy) fourth
   - Execution Control plan/apply fifth
3. Preserve `CoreStepResult` as the public output contract.
4. Add or update tests in `tests/semantics/test_core_pipeline_clean.py`.
5. Confirm quickstart behavior still reflects the public contract.

Guardrails:

- No Runtime dispatch logic in Core pipeline code.
- No legacy compatibility contract restoration.
- Keep deterministic behavior and public API coherence.
