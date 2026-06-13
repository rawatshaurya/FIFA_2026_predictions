## Summary

Describe the behavior changed and why.

## Validation

- [ ] `uv run --frozen pytest --cov=fifa_predict`
- [ ] CLI or notebook smoke test, when applicable
- [ ] No generated datasets, model binaries, prediction artifacts, or secrets added

## Modeling and Data Safety

- [ ] Features use only information available before kickoff
- [ ] Network behavior has a cached or offline path
- [ ] Model changes include chronological validation evidence
- [ ] New third-party data includes source and license documentation
