# results/

One JSON per benchmark run, written by `python -m bench <backend> --label <name>`.
These are the evidence; `docs/RESULTS.md` is only the summary generated from them.

Each file carries the full run: board state (`env` — power mode, clocks, L4T, package
versions, repo sha), backend configuration (`meta`), latency distribution and
per-stage breakdown, the system and per-process monitor windows, the control-loop
figures, and representative action chunks (`saved_chunks`).

Failed runs are written too, with `status: "failed"` and the traceback. A backend that
does not work on this board is a result — do not delete it.

`evo1-bootstrap.ort.json` is infrastructure evidence only. Its embedded metadata marks
the random action head `deployable: false` and records native-fixture parity; its saved
actions must never be sent to a robot.

```bash
python -m bench report results --out docs/RESULTS.md
```
