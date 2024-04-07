# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import re
import os.path as osp

def parse_int(s):
    for i,c in enumerate(s):
        if c not in "0123456789":
            #return int(s[:i]), s[i:]
            return s[:i], s[i:]
    return int(s), ""

def parse_brackets(s):
    # parse a "bracket" expression (including closing ']')
    lst = []
    while len(s) > 0:
        if s[0] == ',':
            s = s[1:]
            continue
        if s[0] == ']':
            return lst, s[1:]
        a, s = parse_int(s)
        assert len(s) > 0, f"Missing closing ']'"
        if s[0] in ',]':
            lst.append(a)
        elif s[0] == '-':
            b, s = parse_int(s[1:])
            lst.extend(range(int(a),int(b)+1))
    assert len(s) > 0, f"Missing closing ']'"

def parse_node(s):
    # parse a "node" expression
    for i,c in enumerate(s):
        if c == ',': # name,...
            return [ s[:i] ], s[i+1:]
        if c == '[': # name[v],...
            b, rest = parse_brackets(s[i+1:])
            if len(rest) > 0:
                assert rest[0] == ',', f"Expected comma after brackets in {s[i:]}"
                rest = rest[1:]
            return [s[:i]+str(z) for z in b], rest

    return [ s ], ""

def parse_list(s):
    lst = []
    while len(s) > 0:
        v, s = parse_node(s)
        lst.extend(v)
    return lst


def setup_dist(world_size: int):

    node_list = parse_list(os.environ.get("SLURM_NODELIST", ""))
    print(node_list)
    os.environ["SLURM_NODELIST"] = ",".join(node_list)
    return node_list[0]

    def extract_first_hostname(slurm_node_list):
        # Check for the simple case where there is no range
        # if "[" not in slurm_node_list and "]" not in slurm_node_list:
        #     return slurm_node_list
        # print(f"SLURM_JOB_NODELIST={slurm_node_list}")
        # # Extract the base part of the node name and the range
        # base_part, range_part = re.match(r"([a-zA-Z0-9-]+)\[([0-9,-]+)\]", slurm_node_list).groups()
        
        # # Handle ranges and individual numbers (split by comma first)
        # first_range = range_part.split(",")[0]
        # if "-" in first_range:
        #     start = first_range.split("-")[0]
        # else:
        #     start = first_range
        # # Reconstruct the first hostname
        # first_hostname = f"{base_part}{start}"
        # return first_hostname     
        pass       
    
    #extract_first_hostname("mojo-26l-r202u[45,47]")
    # os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
    # os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
    # os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", str(world_size))
    # os.environ["MASTER_ADDR"] = extract_first_hostname(os.environ.get("SLURM_JOB_NODELIST", "0.0.0.0"))
    # os.environ["MASTER_PORT"] = "26983"
    # #os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    # print(f"Master: {os.environ['MASTER_ADDR']}")
    # for i in range(1, world_size):
    #     if not os.fork():
    #         os.environ["RANK"] = str(i)

if "SLURM_NODELIST" in os.environ:
    setup_dist(world_size=1)

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local-rank', "--local_rank", type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
