# sort import instructions
import yaml
from pathlib import Path

def load_params():
    params_path = Path("params.yaml")
    with open(params_path, "r", encoding="utf-8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["build_features"]
        except yaml.YAMLError as exc:
            print(exc)