# WeakKC baseline adaptation

`examples/train_weakkc_baseline.py` adds the same diagnostic head to DKT,
DKT+, DKVMN, SAKT, SAINT, AKT, simpleKT, DTransformer, DIMKT and MoCKT.

Protocol is fixed to a 70% history / 30% future split. A KC needs at least ten
future answers; error rate >= 0.60 is positive and <= 0.40 is negative. Future
answers create labels only. Recall@1/3/5/10 is macro-averaged over students
with at least one positive KC.

Multi-KC questions contribute one label observation to each distinct atomic KC;
duplicates inside a question are removed. They remain one question in the
question-level source. Native pyKT baselines retain their official atomic-stream
history encoding so their architectures and pretrained checkpoints remain
comparable. The proposed model uses masked-mean multi-KC embeddings.

The runner starts from a normal pyKT best checkpoint and jointly updates the
encoder and a model-specific instance of the shared head. It interleaves one
WeakKC batch every four KT batches, uses validation KT AUC for early stopping,
and evaluates test data only after selecting the checkpoint.

Before training on another server, run `--check-only`; this constructs all
datasets and executes one diagnostic forward pass without an optimizer step.
Outputs include the combined checkpoint, metrics JSON, standard test prediction
file, WeakKC prediction CSV and summary CSV.
