# Tag The Release Candidate

Tag only after the release commit is pushed, CI is green, the RunPod real smoke evidence is available, and the worktree contains no unintended tracked or staged files.

```bash
git status
git tag -a v0.1.0-rc1 -m "Virtual Try-On demo release candidate"
git push origin v0.1.0-rc1
```

Create the GitHub release from tag `v0.1.0-rc1` and use the contents of `docs/release_notes_v0.1.0-rc1.md` as the release notes.

The tag is intentionally a manual action. `scripts/release_check.sh` validates the candidate but never pushes commits or creates tags.
