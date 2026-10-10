# Evaluation changelog

Holdout cases are frozen by `datasets/<name>/holdout.lock`. Any change to a
holdout case (only to fix a labelling error, never to make a run look better) is
listed here with the reason, and the lock is regenerated in the same commit.

## default 1.0.0 (2026-10-10)

First dataset: 135 cases in 11 categories (108 English, 27 Georgian; 100 EV,
35 tax), 93 dev and 42 holdout. Targets corpus 1.1.0 and its 29 registered traps.
Written before any evaluation run of the agent.
