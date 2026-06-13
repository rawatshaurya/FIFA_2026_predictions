# Security Policy

## Reporting a Vulnerability

Please report vulnerabilities privately through GitHub's Security Advisories
for this repository. Do not include API tokens, private datasets, or other
secrets in a public issue.

This project reads an optional `FOOTBALL_DATA_API_TOKEN` from the environment.
Tokens must remain local and must never be committed. If a token is exposed,
revoke it with the provider before reporting the incident.

Only the latest version on the default branch is supported.
