# results/

One JSON per benchmark run, written by `python -m bench <backend> --label <name>`.
These are the evidence; `docs/RESULTS.md` is only the summary generated from them.

Each file carries the full run: board state (`env` — power mode, clocks, L4T, package
versions, repo sha), backend configuration (`meta`), latency distribution and
per-stage breakdown, the system and per-process monitor windows, the control-loop
figures, and representative action chunks (`saved_chunks`).

Failed runs are written too, with `status: "failed"` and the traceback. A backend that
does not work on this board is a result — do not delete it.

```bash
python -m bench report results --out docs/RESULTS.md
```
