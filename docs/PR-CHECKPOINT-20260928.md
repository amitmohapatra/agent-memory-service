# PR checkpoint, 28 September 2026

The user requested an immediate separate PR to preserve all work with 1% remaining. This is a draft checkpoint, not a release or model-default promotion. No production DB migration or merge was performed.

## Validation

Full offline suite: 1,561 passed, 19 skipped, one failure, 12 deselected. Failure: rebuild fidelity baseline hit the 150 ms graph wall budget on the loaded host. The functional test now has a five-second test-only graph budget and requires identical complete results; production keeps 150 ms. The follow-up failure/deadline suite passed 23 cases, including the corrected rebuild, but failed the worker-kill/requeue test. That failure remains to investigate; do not mark the suite green. Ruff passed the focused edits.

The first real OCR image lacked headless OpenCV and all 24 pages failed. Its artifact is retained as an environment failure. The corrected image imported cv2 successfully, but its OCR run was stopped at the user's checkpoint request. Check artifact image digests: an older completed failure artifact is not a successful corrected run. The owned follow-up queue was stopped before snapshotting. Full fresh LoCoMo arms and the multilingual fusion screen have not completed.

## Remaining work

1. Investigate worker-kill/requeue failure, rerun the affected checks sequentially without competing OCR load.
2. Complete corrected 12-language OCR and fixed multilingual fusion screen; preserved orchestration scripts are under `benchmark/experiments/checkpoint_20260928/`. Adapt workstation paths and historical PIDs before reuse.
3. Run fresh isolated LoCoMo arms (Granite graph off/on, Bekko a8m graph on). No benchmark DB may be reused as another arm.
4. Decide model defaults from evidence. Runtime named-vector ensemble is not implemented. Current default encoder/NLI remain English; this checkpoint does not certify a completely multilingual production stack.
5. No fresh reader answer accuracy, 90% LoCoMo result, or whole-service 20 RPS on 8 vCPU/16 GiB has been established. Do not attribute component results to the deployed application.

See `CPU-MULTILINGUAL-DECISION-20260928.md`, `MULTILINGUAL-IMPLEMENTATION-20260927.md`, and `HINDSIGHT-CAPABILITY-STATUS-20260927.md`. Model weights, caches, secrets, virtual environments and local DB files are excluded from Git. Historical excluded-model results remain evidence only; they do not select Chinese-origin runtime models.
