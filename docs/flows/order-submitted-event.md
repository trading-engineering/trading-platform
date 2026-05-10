# OrderSubmittedEvent

## Emission rule

Runtime emits `OrderSubmittedEvent` only after a successful external `NEW`
dispatch.

## Non-emission cases

- Failed external `NEW` dispatch -> no `OrderSubmittedEvent`.
- Non-`NEW` commands (for example replace/cancel) -> no new
  `OrderSubmittedEvent`.

## Ordering rule in current MVP

`OrderSubmittedEvent` is processed before `mark_intent_sent` on successful
`NEW` dispatch handling.

## Architectural meaning

- `OrderSubmittedEvent` is canonical.
- It represents canonical order-entry confirmation at dispatch success boundary.
- It does not claim full post-submission lifecycle canonicalization.
