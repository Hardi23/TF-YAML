import os
import warnings
from pathlib import Path
import pkg_resources

os.environ['HYDRA_FULL_ERROR'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
REQUIREMENTS_FILE = Path(os.path.join(os.getcwd(), "config/requirements/requirements.txt"))


def check_requirements() -> bool:
    requirements = pkg_resources.parse_requirements(REQUIREMENTS_FILE.open("r"))
    if not requirements:
        warnings.warn("Error parsing requirements, continuing!")
        return True
    else:
        for requirement in requirements:
            try:
                pkg_resources.require(str(requirement))
            except pkg_resources.DistributionNotFound:
                input_str = input(f"Package {str(requirement)} not found, install now?")
                if input_str.lower() in ["y", "yes", "j"]:
                    os.system(f'pip install {str(requirement)}')
                else:
                    return False
        return True

