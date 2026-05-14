# Changelog

## [Unreleased]

### Changed

- Phase R1 clean cut:
  - removed compatibility gate and snapshot lifecycle contracts from Core APIs
  - made `RiskEngine` policy-only (`evaluate_policy_intent`, constraints build)
  - simplified docs/tests to one clean CoreStep/CoreWakeupStep architecture
- Phase R2 polish:
  - removed JSON schema files in favor of Pydantic contract source of truth
  - removed snapshot-shaped execution feedback payload rows
  - renamed constraint normalizer to `ExecutionConstraintsPolicy`
  - hardened clean-pipeline semantics tests (reconciliation, rejection, deferral)

### Added

- clean public exports for canonical events, `ExecutionControl`, and `NullEventBus`
- focused semantics tests for clean Core pipeline and API boundary
