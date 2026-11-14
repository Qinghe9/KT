# pyKT-toolkit integration overlay

This directory contains only the files changed or added for the unified baseline protocol.

- Upstream: <https://github.com/pykt-team/pykt-toolkit>
- Base commit: `77c3e90fdb807542194b989656ccac10e5d92e12`
- Purpose: 9-fold student split with a 10% held-out test set, non-window evaluation, atomic-KC prediction export, Macro-KCAUC, and a shared WeakKC diagnostic head.

Apply the overlay to a clean upstream checkout:

```bash
git clone https://github.com/pykt-team/pykt-toolkit.git
cd pykt-toolkit
git checkout 77c3e90fdb807542194b989656ccac10e5d92e12
cp -a /path/to/KT/integrations/pykt-toolkit-overlay/. ./
pip install -e .
```

The overlay intentionally excludes raw datasets, checkpoints and generated predictions. See `docs/EXPERIMENT_GUIDE.md` in the main repository for the end-to-end protocol.
