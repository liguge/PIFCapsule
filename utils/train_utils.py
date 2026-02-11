#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import torch
from torch import optim
import models
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from utils.losses import TotalLoss
import torch.nn.functional as F



class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        if args.processing_type == 'O_A':
         print(1)
        elif args.processing_type == 'R_NA':
            from CNN_Datasets.R_NA import datasets
            Dataset = getattr(datasets, args.data_name)
        else:
            raise Exception("processing type not implement")

        print(Dataset)

        self.datasets = {}

        self.datasets['train'], self.datasets['val'] = Dataset(args.data_dir,args.normlizetype).data_preprare()

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                           batch_size=args.batch_size if x == 'train' else 4*args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        # Define the model
        self.model = getattr(models, args.model_name)(num_class=args.num_class)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            conv1_weight = self.model.module.efficient_capsnet.conv1.weight
            conv2_weight = self.model.module.efficient_capsnet.conv2.weight

            # 定义优化器
            other_params = [
                param for name, param in self.model.named_parameters()
                if param is not conv1_weight and param is not conv2_weight
            ]

            # 定义优化器
            self.optimizer = optim.AdamW([
                {'params': conv1_weight, 'lr': 0.001},  # conv1.weight 学习率设置为 0.1
                {'params': conv2_weight, 'lr': 0.001},  # conv2.weight 学习率设置为 0.1
                {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay}  # 其他参数使用默认学习率
            ])
        else:
            raise Exception("optimizer not implement")
        self.best = 0
        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Load the checkpoint
        self.start_epoch = 0

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = TotalLoss(recon_factor=0.0005, bd_factor=0.0005)  #胶囊网络 0.0005



    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()
        losses = []
        val_losses = []
        acces = []
        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with (torch.set_grad_enabled(phase == 'train')):


###########################################################################################################
                        out_inputs, out_labels, loss2, g = self.model(inputs, mode=phase)
                        margin, recon, bd_loss = self.criterion(inputs, F.one_hot(labels, args.num_class), out_inputs, out_labels)
                        loss = self.model.module.UW(1 * loss2.sum(), 1e4 * g.sum(), recon/10., bd_loss/100.) + margin
                        pred = out_labels.argmax(dim=1)

                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0*batch_count/train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx*len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # Print the train and val information via each epoch
                if phase == 'train':
                    epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                    epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                    train_epoch_loss = epoch_loss
                else:
                    epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                    epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                    val_epoch_loss = epoch_loss
                    val_epoch_acc = epoch_acc
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time()-epoch_start
                ))



                # save the model
                if phase == 'val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc:  #or epoch > args.max_epoch-2:
                        best_acc = epoch_acc
                        self.best = epoch
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))


            if self.lr_scheduler is not None:
               self.lr_scheduler.step()


            acces.append(val_epoch_acc)
            losses.append(train_epoch_loss)
            val_losses.append(val_epoch_loss)

        pd.set_option('display.max_columns', None)  # 显示完整的列
        pd.set_option('display.max_rows', None)  # 显示完整的行
        dataframe = pd.DataFrame({'losses': losses, 'val_losses': val_losses, 'acces': acces})
        csv_file_path = os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)) + '.csv'
        dataframe.to_csv(csv_file_path, index=False, sep=',')
        # 读取生成的 CSV 文件
        df = pd.read_csv(csv_file_path)
        # 获取 'acces' 列并排序，取前五个最大的值
        top_5_acces = df['acces'].nlargest(5)
        # 计算平均值和方差
        mean_value = top_5_acces.mean()
        variance_value = top_5_acces.var()
        # 输出结果
        logging.info(f"前五个最大值的平均值: {mean_value}")
        logging.info(f"前五个最大值的方差: {variance_value}")



    def test(self):
        num_class = self.args.num_class   ###类别数目
        reals = []
        pres = []
        logitss = torch.zeros(4*self.args.batch_size, num_class).cuda()
        with torch.no_grad():
            self.model.eval()
            for filename in os.listdir(self.save_dir):
                # 检查文件名中是否包含"init"
                if (str(self.best) + '-0') in filename or (str(self.best)+'-1') in filename:
                    if self.device_count > 1:
                        self.model.module.load_state_dict(torch.load(os.path.join(self.save_dir, filename), weights_only=False))
                        break
                    else:
                        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, filename), weights_only=True))
            for batch_idx, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ###############################
                _, logits,_,_ = self.model(inputs)
                pred = logits.argmax(dim=1)
                reals.extend(labels.cpu().detach().numpy())
                pres.extend(pred.cpu().detach().numpy())
                logitss = torch.cat((logitss, logits))
        y_test = np.array(reals).reshape(1, -1).squeeze()
        yh_test = np.array(pres).reshape(1, -1).squeeze()
        logging.info('Shape of y_test: %s', y_test.shape)
        logging.info('Shape of yh_test: %s', yh_test.shape)
        logging.info('Number of matches: %d', sum(y == yh for y, yh in zip(y_test, yh_test)))
        logging.info('Accuracy score: %f', accuracy_score(y_test, yh_test))
        return logitss[4*self.args.batch_size:, :].cpu().detach().numpy(), y_test













