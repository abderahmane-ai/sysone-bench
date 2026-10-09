# Session Handoff - 2026-10-03

State at the end of the 2x T4 session that ended on a Kaggle weekly quota reset. Written so a
fresh session can resume without re-deriving anything.

## Safe data: 24 complete runs

Verified locally against their own `checksums.sha256` at handoff time: **24 verified, 0 failed.**

Copy locations:
- Laptop: `~/sysone-bench-results/cpu/runs/` (15 CPU) and `~/sysone-bench-results/gpu/` (9 GPU)
- Hugging Face private dataset `saidutta69/sysone-bench-gpu-results`, head `4882402a021d`, holds the
  same 9 GPU runs. This is the remote backup; the Kaggle disk is not.

CPU 15: `bosun-06b` `decision-eos` `decision-kai` `decision-lex` `decision-sol` `gliner-base`
`gliner-decide` `gliner-multi` `gliner-small` `laya` `lumma-fev-01b` `lumma-fev-06b` `mojev`
`tev1-08b` `this-that-12`

GPU 9: `bosun-17b` `decision-nox` `intern-decision-08b` `intern-decision-2b` `intern-decision-4b`
`jpt-08b` `julia-1` `lavoir` `tev1-4b`

## Lost: scores known, artifacts gone

These completed on Kaggle but the session died before the HF upload step, so only the scores
survive. All are reproducible from the sealed manifest and the pinned adapters in this repo.

| Model | Score | Cause of loss |
| :--- | ---: | :--- |
| `jevk5` | 0.8508 | weekly quota reset before upload |
| `decider-4b` | 0.8411 | earlier session death, disk full |
| `neohorse-4b` | 0.8266 | same |
| `decider-2b` | 0.7927 | ran on Kaggle and Colab; Colab copy partially pulled |
| `kev-08b` | 0.7726 | earlier session death, disk full |

`jevk5` at 0.8508 is the most valuable loss: it was the highest single score of the session.

Partial salvage for `decider-2b`: `~/sysone-bench-results/colab/` holds its Colab
`summary.json`, `metadata.json` and `checksums.sha256`, but **not** `predictions.jsonl`, so it is not
a complete run record and must not be counted as one.

## Registered adapters, all tested

23 vendor runners plus the Laya baseline. `tests/test_vendor_common.py` passes 56/56 in an
environment with the vendor SDKs present. Locally one test fails
(`test_thisthat_answers_every_question_in_one_call_and_keeps_declared_labels`) purely because the
`thisthat` package is not installed on the laptop; it passes in a prepared venv.

Never successfully run, adapter written and ready: `jet`.

## Known issues to fix first

1. **`kev-08b` regression.** It scored 0.7726 on an earlier environment and then failed in 57s on a
   clean one, after the fp32-to-bf16 fallback was added to `runners/vendor/kev.py`. Cause not yet
   diagnosed. The fallback tries fp32 first, and on server-start failure retries bf16; it may be
   retrying a failure that is not VRAM.
2. **`kev-4b` needs bf16.** fp32 is the card's stated evaluation path but needs roughly 18.6 GB for a
   4B model against 14.56 GiB of T4 VRAM, so it OOMs. The fallback records `precision_deviation` so
   the row is never mistaken for the card's exact path.
3. **Never queue two workers on one card.** A duplicate queue on cuda:1 was created by mistake and
   the kill could not be confirmed before the session ended. `gpuq.sh` does not prevent this; add a
   lock file per card if more parallel queues are ever run.

## Storage guard, now built in

The session that died did so because the volume filled and the filesystem remounted read-only. The
queue script `gpuq.sh` now, per model:

- refuses to start a download below 40 GB free, after first sweeping the shared HF blob store;
- sweeps blobs between models. HF dedupes weights into one `blobs/` directory and each model's
  `snapshots/` symlinks into it, so deleting a snapshot frees nothing. Only blobs still referenced
  by a live snapshot may be kept;
- uploads the finished run to the HF private dataset before starting the next model.

Any new session must reuse `gpuq.sh` rather than an ad-hoc loop.

## Resuming

1. Recreate the Kaggle T4 session and the `zrok` tunnel on port 9191.
2. Push the working tree, then `/root/setup2.sh` rebuilds the venv, installs the SDKs, writes
   `gpuq.sh` and verifies the manifest digest.
3. `chmod +x /root/gpu_harness.sh /root/gpuq.sh` - `scp` does not preserve the executable bit, and
   omitting this yields `rc=126` with every model failing instantly.
4. Launch at most one queue per card, for example:
   `nohup /root/gpuq.sh 0 c0 decider-2b kev-08b decider-4b neohorse-4b &`
   `nohup /root/gpuq.sh 1 c1 kev-4b kev-9b jevk5 &`
5. Re-run the five lost rows before starting new adapter work.

## Remaining scope

Target is 53 models; 24 are safe. Registered and runnable without new code: `decider-2b`,
`decider-4b`, `kev-08b`, `kev-4b`, `kev-9b`, `jevk5`, `jpt-9b`, `neohorse-4b`.

Needing an adapter, in rough order of value: `semif` and `jevk5` share a letter-logit idea but not an
API, so `semif` is separate work; then `hopper-g`, `metask`, `pngwn`, `lev`, `winnow-e4b`, `jet`,
`lfm350`, `lfm2600`, `verdict`.

Unreachable: `akash-gemma` only. See `decision-index-panel-coverage.md`.

## Repo state

Nothing has been committed. All adapter, test and documentation work is uncommitted in the local
working tree. Nothing was pushed to `master`, no PR was merged, and nothing was published.