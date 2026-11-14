"""Run the selected ten EdNet baselines with the shared local reporter."""

import sys

from run_ednet_baselines import main


DEFAULT_MODELS = [
    "dkt", "dkt+", "dkvmn", "sakt", "saint", "akt", "simplekt",
    "dtransformer", "dimkt", "mockt",
]


if __name__ == "__main__":
    if "--dataset-name" not in sys.argv:
        sys.argv.extend(["--dataset-name", "ednet"])
    if "--models" not in sys.argv:
        sys.argv.extend(["--models", *DEFAULT_MODELS])
    main()
