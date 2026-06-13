# Contributing

Thanks for helping improve FIFA Predict.

## Development Setup

The project uses Python 3.12 and uv:

```bash
uv sync --locked --dev
uv run --frozen pytest
```

Keep generated datasets, caches, trained models, prediction exports, API
tokens, and personal environment files out of commits. The existing
`.gitignore` covers the standard local paths.

## Pull Requests

1. Keep changes focused and explain the user-visible behavior.
2. Add or update tests for behavioral changes.
3. Preserve strict point-in-time feature construction. A match may not use its
   own score, same-day later results, future availability reports, or future
   lineup announcements.
4. Keep network access optional. Tests should use local fixtures or mocked
   responses.
5. Run `uv run --frozen pytest --cov=fifa_predict` before opening the PR.

For model changes, include rolling-origin log loss and the relevant supporting
metrics. A more complex model should not replace a simpler model unless the
chronological validation result improves.

## Data Contributions

Do not commit third-party bulk datasets unless their license explicitly permits
redistribution. Prefer ingestion code, source links, small synthetic fixtures,
and documented schemas. Never commit `FOOTBALL_DATA_API_TOKEN`.

By contributing, you agree that your work will be licensed under the MIT
License.
