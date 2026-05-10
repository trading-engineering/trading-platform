# Core Pipeline Map

This page maps CoreStep/CoreWakeupStep processing from concept to code.

Runtime provides canonical inputs and later dispatches results. Core performs deterministic
reduction and decision shaping, then returns `CoreStepResult`.

## `run_core_step` pipeline

1. Canonical `EventStreamEntry` input  
   - Responsibility: accept positioned canonical input from runtime.  
   - Key models/functions: `EventStreamEntry`, `ProcessingPosition`, `run_core_step`.  
   - Output: one step invocation context.

2. State reduction  
   - Responsibility: reduce canonical event into `StrategyState`.  
   - Key functions: `process_event_entry`, `process_canonical_event`.  
   - Output: updated state snapshot.

3. Strategy evaluation  
   - Responsibility: optionally evaluate strategy for generated intents.  
   - Key models/functions: `CoreStepStrategyEvaluator`, `CoreStepStrategyContext`.  
   - Output: `generated_intents`.

4. Generated + queued candidate combination  
   - Responsibility: merge generated and queued intents deterministically.  
   - Key functions/models: `combine_candidate_intent_records`, `CandidateIntentRecord`.  
   - Output: `candidate_intent_records` and derived `candidate_intents`.

5. Dominance / reconciliation  
   - Responsibility: resolve effective intent set by deterministic dominance rules.  
   - Key location: `intent_combination.py`.  
   - Output: effective candidate ordering for policy/execution stages.

6. Policy admission  
   - Responsibility: side-effect-safe policy filtering for generated-origin candidates.  
   - Key functions/models: `apply_policy_to_candidate_records`, `PolicyRiskDecision`.  
   - Output: accepted/rejected policy projection.

7. Execution-control planning/apply  
   - Responsibility: shape and optionally apply dispatchability mechanics.  
   - Key functions/models: `plan_execution_control_candidates`, `apply_execution_control_plan`,
     `ExecutionControlDecision`.  
   - Output: queued/dispatchable/handled projections and optional scheduling obligation.

8. Dispatchable output  
   - Responsibility: expose runtime-consumable dispatch set.  
   - Key model: `CoreStepResult.dispatchable_intents`.  
   - Output: dispatch candidates for runtime-owned execution.

9. Control scheduling output  
   - Responsibility: return non-canonical scheduling handoff for runtime realization.  
   - Key model: `CoreStepResult.control_scheduling_obligation`.  
   - Output: optional `ControlSchedulingObligation`.

## `run_core_wakeup_step` pipeline

- `run_core_wakeup_step` is a convenience wrapper composed of:
  1. `run_core_wakeup_reduction` (reduce multiple entries + collect generated intents)
  2. `run_core_wakeup_decision` (candidate merge, policy admission, optional execution-control apply)
- It returns `CoreStepResult` with the same dispatchability/scheduling output shape used by
  single-step processing.

## Runtime boundary

Runtime dispatch happens after Core returns. Core does not perform external venue dispatch itself.
