import argparse

from train import train

import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,
                        help='name of this task: train/generate', required=True)
    return parser.parse_args()
    
if __name__ == '__main__':

    args = parse_arguments()

    if args.task == 'train':
        train()
    else:
        raise ValueError('Invalid Task')