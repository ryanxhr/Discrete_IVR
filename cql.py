import argparse
import d3rlpy
import wandb
import numpy as np

from d3rlpy.algos import IQL, DiscreteCQL, DiscreteSAC
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

import os 
# os.system('export WANDB_API_KEY=9d45bb78a65fb0f3b0402a9eae36ed832ae8cbdc')
# os.system('wandb login')
def main(args):
    # export WANDB_KEY_API=
    name = f'CQL_{args.dataset}'
    wandb.init(project='atari',
               entity='louis_t0',
               name=name,
               config={
                'alg': 'CQL',
                'env': args.dataset,
                'ratio': args.ratio
               })
    dataset, env = get_atari(args.dataset)
    
    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=1-args.ratio)

    cql = DiscreteCQL(
        n_frames=4,  # frame stacking
        q_func_factory=args.q_func,
        scaler='pixel',
        use_gpu=args.gpu)

    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=200,
            scorers={
                'environment': evaluate_on_environment(env, epsilon=0.05)
                # 'td_error': td_error_scorer,
                # 'discounted_advantage': discounted_sum_of_advantage_scorer,
                # 'value_scale': average_value_estimation_scorer
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='breakout-mixed-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--q-func',
                        type=str,
                        default='mean',
                        choices=['mean', 'qr', 'iqn', 'fqf'])
    parser.add_argument('--gpu', default=0,type=int)
    args = parser.parse_args()
    main(args)