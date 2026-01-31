# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
from mmcv import Config

from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('--config', type=str, default='./configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py', help='test config file path')
    parser.add_argument('--result', type=str, default='./output_debug/result.pkl', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', type=str, default='./show-dir', help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    # build the dataset
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)

    if getattr(dataset, 'show', None) is not None:
        # data loading pipeline for showing
        eval_pipeline = cfg.get('eval_pipeline', {})
        if eval_pipeline:
            dataset.show(results, args.show_dir, pipeline=eval_pipeline)
        else:
            dataset.show(results, args.show_dir)  # use default pipeline
    else:
        raise NotImplementedError(
            'Show is not implemented for dataset {}!'.format(
                type(dataset).__name__))


if __name__ == '__main__':
    main()
