# Public release scope

This update publishes experiment code and de-identified reproducibility records. The author explicitly requested that manuscript files remain private.

Included:

- The curated training, generation, teacher-labeling, detection, statistics, audit and figure-source scripts.
- Three diffusion orchestration scripts, with only workstation/interpreter paths made portable.
- The updated evaluation-workflow figure source.
- The reproducibility archive, its full checksum index, and its internal checksum validator.
- Archived results, fixed content-hash identifiers and configuration snapshots already contained in that archive.

Not included:

- Chinese or English manuscripts, manuscript revision scripts, author/title pages or reviewer correspondence.
- Raw images, generated image pools, original private image filenames or model weights.
- Third-party full-text papers, credentials, caches or training logs.

The repository is not a full backup of the author's workstation. Historical and obsolete experiment pipelines are not silently substituted for the final protocol. Original numerical records in the reproducibility archive are preserved, including negative and nonsignificant results. Privacy screening is a pattern-based check, not a guarantee of anonymization against every possible external linkage.

Source files are directly browsable and are the canonical code copy. The redundant root-level code ZIP was removed after all 72 archive members were verified as byte-identical to the source tree. Use GitHub's **Code → Download ZIP** or `git clone` for the current version; the former ZIP remains recoverable from Git history. The separate numerical reproducibility archive is retained unchanged. This is not an end-to-end retraining or a claim that code-only access is sufficient to reproduce every numerical result. Consult `DATA.md`, `PROTOCOL.md` and `THIRD_PARTY.md` before running experiments.

## Verification without training

```bash
python tools/check_release.py
python tools/check_supplement.py
```

Neither command imports training modules or launches GPU work. The supplement checker validates all files against the archive's `INDEX.json` and the archive checksum in `reproducibility/ARCHIVE.json`.

The publication manifest records source/release hashes. Portability edits are restricted to release copies; the original experiment files, trained weights, losses, model architectures, numerical results and private manuscripts are not modified by this upload.
