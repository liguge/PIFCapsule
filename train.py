#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
from utils.logger import setlogger
from utils.train_utils import train_utils
from lightning.pytorch import seed_everything

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # basic parameters
    parser.add_argument('--model_name', type=str, default='facnet', help='the name of the model')  #facnet
    parser.add_argument('--data_name', type=str, default='G', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default= "E:\he_fault_diagnosis\paper5\DL-based-Intelligent-Diagnosis-Benchmark-master\DataPaper5\G\\G_200.npy", help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '-1-1', 'mean-std'], default='mean-std', help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='R_NA',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=256, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--num_class', type=int, default=9, help='the number of class')
    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='100', help='the learning rate decay for step and stepLR')
    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import logging, datetime
    from datetime import datetime, timezone, timedelta
    seed_everything(3407)
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    print(args.cuda_device.strip())
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # Prepare the saving path for the model
    sub_dir = args.model_name+'_'+args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # set the logger
    setlogger(os.path.join(save_dir, datetime.now().astimezone(timezone(timedelta(hours=8))).strftime("beijing%Y%m%d%H%M%S.txt")))
    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
    trainer.tsne()








