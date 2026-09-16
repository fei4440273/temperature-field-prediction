"""Retired LOGO/grouped-CV API.

The project now uses one immutable train/validation/test protocol. These names
remain only so older callers fail with an explicit migration message.
"""


def _disabled(*_args, **_kwargs):
    raise RuntimeError(
        "LOGO and grouped cross-validation are disabled; use the fixed train/validation/test protocol"
    )


write_logo_protocol = _disabled
aggregate_logo_runs = _disabled
aggregate_logo_seeds = _disabled
write_grouped_cv_protocol = _disabled
aggregate_grouped_cv_runs = _disabled
aggregate_grouped_cv_seeds = _disabled
