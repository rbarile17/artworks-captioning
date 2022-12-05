"""
Wrapper module used to launch the training script.
"""

import os
from pathlib import Path
from ... import load_params
from ...data import dataset_paths

def main(params=None):
    """
    Main method.
    """
    if params is None:
        params = load_params()["train_AdaIn"]

    # set up parameters
    args = {}

    # mandatory parameters
    args['--content_dir'] = dataset_paths[params.pop('content_dataset')]
    args['--style_dir'] = dataset_paths[params.pop('style_dataset')]

    # optional parameters with default in the original script
    if 'style_filter' in params:
        args['--style_filter'] = dataset_paths[params.pop('style_filter')]

    for param, value in params.items():
        args[f"--{param}"] = value
    
    # optional parameters with custom default
    if "vgg" not in args:
        args["--vgg"] = "./models/AdaIn/input/vgg_normalised.pth"

    # start training
    ADAIN_PATH = Path(os.environ["ADAIN_PATH"])
    train_script = ADAIN_PATH / "train.py"
    command = f"python {train_script}"
    for key, value in args.items():
        if value is None or not value:
            continue
        command += f" {key} {value}"
    os.system(command)


if __name__ == "__main__":
    main()
