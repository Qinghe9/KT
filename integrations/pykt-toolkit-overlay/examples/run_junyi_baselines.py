"""Convenience entry point for the Junyi2015 AKT/simpleKT/QIKT baselines."""

import sys

from run_ednet_baselines import main


if __name__ == "__main__":
    if "--dataset-name" not in sys.argv:
        sys.argv.extend(["--dataset-name", "junyi2015"])
    if "--models" not in sys.argv:
        sys.argv.extend(["--models", "akt", "simplekt", "qikt"])
    main()
