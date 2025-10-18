
## Prompt 8 — Post-Launch Checklist
```
Assume the release has shipped. Execute the post-launch checklist:
1. Tag the release (`git tag -a v0.2.0 -m "scPerturb-CMap 0.2.0 GA"`), but push only after QA sign-off.
2. Archive demo outputs in long-term storage if required (document retention location).
3. Create GitHub issues/milestones for:
   - Future authentication/rate-limiting,
   - Expanded readiness checks (Redis/Postgres live endpoints),
   - Docs automation (codespell, acceptance in CI).
4. Schedule a retrospective meeting; jot down agenda topics in `docs/guides/CHANGELOG.md` or a dedicated `RETRO.md`.
5. Confirm all validation artifacts (logs, screenshots) are committed or safely stored.
```

---
