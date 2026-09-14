# PUBLIC-RESEARCH-REFRESH-002: final Q3_PLE research refresh

## Definition of done

The public `nickmatteo/Qwen38-FlashNext-Q3PLE-MTP` repository accurately
documents the strongest qualified result and the final daily native-vision
system. It includes the exact runtime delta, public-safe profile and gateway,
compact normalized evidence, and clear reproduction steps. The 31.45 tok/s
short fixture, 20.253 tok/s occupied-60K qualification, and 15.82 tok/s daily
native-vision row remain separate. The diff passes focused tests, public-release
validation, credential/path scans, link checks, and an evidence review.

A pull request must use source branch `docs/public-research-20260914` and target
`main` in the public repository. LinkedIn project copy and a new-post draft use
the same claim boundary. The new post remains a draft until publication is
explicitly approved.

## Dependencies and status

| Step | Status | Validation |
| --- | --- | --- |
| Audit real public repo, lab checkout, and measured evidence | DONE | Public/private remotes distinguished; dirty lab checkout isolated |
| Freeze claims and minimal public file set | DONE | Every number maps to a preserved result; no absolute first claim |
| Add runtime, profile, gateway, evidence, and documentation | DONE | Daily 4-patch and occupied-60K 24-patch deltas; distinct profiles and receipts |
| Run tests, redaction scan, links, and public-release checks | DONE | 130 tests, public validator, changed-file scan, tree reconstruction, card render, and independent evidence review pass |
| Commit, push, and open PR to public `main` | DONE | Publication commit `e5b7da3`; public PR #3 targets `main` |
| Update LinkedIn project | BLOCKED | T3 preview has no authenticated LinkedIn session; Windows Computer Use native pipe is unavailable |
| Prepare LinkedIn post | DONE | Date-bounded draft and 1200 x 627 card prepared; publication remains unapproved |

## Planned validation

```powershell
python -m pytest tests/test_q3ple_daily_profile.py tests/test_q3ple_dsh_gateway.py
python -m py_compile scripts/q3ple_daily_profile.py scripts/q3ple_dsh_gateway.py
python scripts/validate_public_release.py
git diff --check origin/main... -- . ':(exclude)patches/llama.cpp/**/*.patch'
```

Generated email patches are excluded from whitespace lint because they must
preserve the measured source diff byte-for-byte, including source lines that
already contain trailing spaces. Their correctness gate is clean application
plus the expected final tree hash.

Patch reconstruction is also verified from the pinned Cafe base. The base
36-patch series produces source tree `db305f143fefd46da53db8a8d751b882da38274f`;
the daily four-patch delta produces `95417c378aee171d51ac09808c16f0b33b0fbdc0`;
and the occupied-60K 24-patch delta produces
`7b3a784de1922d023ee5f040b19873903b26aeb4`.

The final review must confirm that no API key, credential, private network
hostname, user-profile path, model payload, runtime log, or mutable process
state is included.
