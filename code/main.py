import argparse

from train import train
from generate import generate

import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,
                        help='name of this task: train/generate', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of the model')
    return parser.parse_args()
    
if __name__ == '__main__':

    args = parse_arguments()

    if args.task == 'train':
        train()
    elif args.task == 'generate':
        generate(args)
    else:
        raise ValueError('Invalid Task')