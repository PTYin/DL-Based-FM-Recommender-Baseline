import os
import argparse
import yaml
from src.main import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='config_example/DeepFM_load.yaml',
                        help='path for configure file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("config file doesn't exist")
        exit(-1)
    run(yaml.load(open(args.config, 'r').read(), Loader=yaml.FullLoader))

