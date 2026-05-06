'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * Changed from SLIP
 * https://github.com/facebookresearch/SLIP
 * By Le Xue
'''
# (Possible) workaround for Open3D compatibility with PyTorch multiprocessing
#import torch.multiprocessing as mp
#mp.set_start_method('forkserver', force=True)

import argparse
from collections import OrderedDict
import math
import time
import wandb

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections

from data.dataset_3d import *

from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn
from data.dataset_3d import BalancedBatchSampler
import open_clip


def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='objaverse', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--validate_dataset_name', default='nuscenes_objects', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    
    # Optional second validation set
    parser.add_argument('--validate_dataset_name_2', type=str)
    parser.add_argument('--validate_dataset_prompt_2', type=str)
    
    parser.add_argument('--use_height', action='store_true', help='whether to use height information, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP2_PointBERT', type=str)
    # Training
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float) # changed from (0.9, 0.98)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval ulip only')
    parser.add_argument('--evaluate_3d_ulip2', action='store_true', help='eval ulip2 only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--no-distributed', action='store_true', 
                        help='disable distributed training completely')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb-id', default=None, type=str, help='WandB run ID (optional)')

    parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')
    parser.add_argument('--ckpt_path', default=None, type=str, help='Path to pretrained checkpoint')

    parser.add_argument('--lr-block', default=2e-4, type=float, help='lower LR for encoder “block” layers')
    parser.add_argument('--sim-occlusion', action='store_true', help='use occlusion simulation')
    parser.add_argument('--clip-grad', default=0, type=float, help='gradient clipping tershold; 0 means no clipping')
    parser.add_argument('--linear-projection', action='store_true', help='use linear projection instead of MLP')
    # parser.add_argument('--pooling-type', default='mean', type=str, help='pooling type', choices=['sum', 'mean', 'mix', 'max'])
    parser.add_argument('--max-lidar-ratio', default=0.3, type=float, help='maximum ratio of lidar points in a batch for objects_joint dataset')
    parser.add_argument('--test_repeat', default=1, type=int, help='Number of times to repeat evaluation for statistical analysis')
    parser.add_argument('--static', action='store_true', help='use constant mixing ratio for all batches')
    parser.add_argument('--start-ckpt', action='store_true', help='load weights from resume path but start training from scratch')
    parser.add_argument('--excluded-classes', nargs='*', default=[], type=str, help='space separated list of classes to exclude from training (for nuscenes_objects)')

    return parser

best_acc1 = 0

def main(args):
    utils.init_distributed_mode(args)

    global best_acc1

    if utils.is_main_process() and args.wandb:
        wandb.login(key="KEY")
        wandb_id = args.wandb_id if args.wandb_id else os.path.split(args.output_dir)[-1]
        wandb.init(project='PROJECT', id=wandb_id, config=args, reinit=True, entity='ENTITY', resume="allow")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.evaluate_3d:
        zero_stats = test_zeroshot_3d(args)
        print(zero_stats)
        return
    elif args.evaluate_3d_ulip2:
        zero_stats = test_zeroshot_3d_ulip2(args)
        print(zero_stats)
        return

    # create model
    print("=> creating model: {}".format(args.model))
    model, tokenizer, train_transform, _ = getattr(models, args.model)(args=args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=False)

    # define loss function (criterion) and optimizer
    criterion = models.get_loss(args).cuda(args.gpu)

    #p_wd, p_non_wd = [], []
    p_wd, p_non_wd, p_block_wd, p_block_non_wd, p_logit_scale = [], [], [], [], []
    for n, p in model.named_parameters():
        """
        if not p.requires_grad:
            print('in optimizer freeze {}'.format(n))
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
            (p_block_non_wd if 'block' in n else p_non_wd).append(p)
        else:
            p_wd.append(p)
            (p_block_wd if 'block' in n else p_wd).append(p)
        """
        if not p.requires_grad:
            print(f'in optimizer freeze {n}')
            continue

        is_block = 'block' in n
        is_norm_or_bias = (p.ndim < 2) or ('bias' in n) or ('ln' in n) or ('bn' in n)
        is_logit_scale = 'logit_scale' in n # DO NOT use weight decay for logit scale!!

        if is_logit_scale:
            p_logit_scale.append(p)
        elif is_block:
            if is_norm_or_bias:
                p_block_non_wd.append(p)
            else:
                p_block_wd.append(p)
        else:
            if is_norm_or_bias:
                p_non_wd.append(p)
            else:
                p_wd.append(p)

    #optim_params = [{"params": p_wd, "weight_decay": args.wd},
    #                {"params": p_non_wd, "weight_decay": 0}]
    optim_params = [
        {"params": p_wd,            "weight_decay": args.wd, "lr": args.lr, "block": False},
        {"params": p_non_wd,        "weight_decay": 0,       "lr": args.lr, "block": False},
        {"params": p_block_wd,      "weight_decay": args.wd, "lr": args.lr_block, "block": True},
        {"params": p_block_non_wd,  "weight_decay": 0,       "lr": args.lr_block, "block": True},
        {"params": p_logit_scale,   "weight_decay": 0,       "lr": args.lr, "block": False}
    ]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch if not args.start_ckpt else 0
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint and not args.start_ckpt else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint and not args.start_ckpt else ()
            best_acc1 = checkpoint['best_acc1'] if not args.start_ckpt else 0
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from the latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch'] if not args.start_ckpt else 0
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer']) if not args.start_ckpt else ()
            scaler.load_state_dict(latest_checkpoint['scaler']) if not args.start_ckpt else ()
            best_acc1 = latest_checkpoint['best_acc1'] if not args.start_ckpt else 0
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))
        
    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    if 'ULIP2' in args.model:
        print("=> Using OpenCLIP tokenizer and transform")
    else:
        # Fallback to SimpleTokenizer for original ULIP models
        tokenizer = SimpleTokenizer()
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.ToTensor(),
                normalize
            ])

        print("=> Using SimpleTokenizer and transform")

    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    val_dataset = get_dataset(None, tokenizer, args, 'val')

    if 'objects_joint' in args.pretrain_dataset_name:
        #nuscenes_size = len(train_dataset.nuscenes_datapath)
        objaverse_size = len(train_dataset.objaverse_datapath)
        gpus_num =  utils.get_world_size()
        prob = 0.80
        total_batches = math.ceil((objaverse_size * math.log(1 / (1 - prob))) / gpus_num / args.batch_size) # ref: coupon collector's problem
        
        train_batch_sampler =   DistributedScheduledBatchSampler(dataset=train_dataset,
                                                               batch_size=args.batch_size,
                                                               shuffle=True,
                                                               drop_last=True,
                                                               total_batches=total_batches,
                                                               max_lidar_ratio=args.max_lidar_ratio,
                                                               static=args.static,
                                                               total_epochs=args.epochs,
                                                               warmup_epochs=args.warmup_epochs) \
                if args.distributed else \
                                ScheduledBatchSampler(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           total_batches=total_batches,
                                           max_lidar_ratio=args.max_lidar_ratio,
                                           static=args.static,
                                           total_epochs=args.epochs,
                                           warmup_epochs=args.warmup_epochs)
                 
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler,
            num_workers=args.workers, pin_memory=True,
            collate_fn=customized_collate_fn,
            #multiprocessing_context=torch.multiprocessing.get_context("forkserver"),
            persistent_workers=True if args.workers > 0 else False
            )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
            collate_fn=customized_collate_fn,
            #multiprocessing_context=torch.multiprocessing.get_context("forkserver"),
            persistent_workers=True if args.workers > 0 else False
            )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False,
        collate_fn=customized_collate_fn,
        #multiprocessing_context=torch.multiprocessing.get_context("forkserver"),
        persistent_workers=True if args.workers > 0 else False
        )
    
    if args.validate_dataset_name_2 and args.validate_dataset_prompt_2:
        val_dataset_2 = get_dataset(None, tokenizer, args, 'val_2')
        
        val_sampler_2 = torch.utils.data.distributed.DistributedSampler(val_dataset_2) if args.distributed else None

        val_loader_2 = torch.utils.data.DataLoader(
            val_dataset_2, batch_size=args.batch_size, shuffle=(val_sampler_2 is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler_2, drop_last=False,
            collate_fn=customized_collate_fn,
            #multiprocessing_context=torch.multiprocessing.get_context("forkserver"),
            persistent_workers=True if args.workers > 0 else False
)

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)
    
    ratio = args.lr_block / args.lr
    lr_schedule_low = [v * ratio for v in lr_schedule]

    print(args)

    print("=> beginning training")
    
    global wandb_step
    if utils.is_main_process() and args.wandb:
        # better handles the case when training starts from zero or a non-zero epoch
        iters_per_epoch = len(train_loader) // args.update_freq
        logs_per_epoch =  (iters_per_epoch - 1) // args.print_freq + 1
        wandb_step = args.start_epoch * (logs_per_epoch + 1) # 1 is for the summary log after epoch

    best_epoch = -1

    for epoch in range(args.start_epoch, args.epochs):
        if 'objects_joint' in args.pretrain_dataset_name:
            train_batch_sampler.set_epoch(epoch)

        elif args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, lr_schedule_low, args)
        val_stats = {"acc1": -1}

        if epoch % 1 == 0:

            val_stats = test_zeroshot_3d_core(val_loader, args.validate_dataset_name, args.validate_dataset_prompt, model, tokenizer, args)
            acc1 = val_stats["acc1"]
            print(val_stats)
            
            # validation on the second dataset
            if args.validate_dataset_name_2 and args.validate_dataset_prompt_2:
                val_stats_2 = test_zeroshot_3d_core(val_loader_2, args.validate_dataset_name_2, args.validate_dataset_prompt_2, model, tokenizer, args)
                print(val_stats_2)

            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch

            best_acc1 = max(acc1, best_acc1)

            if is_best or epoch % 10 == 0:
                print("=> saving checkpoint")
                utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc1': best_acc1,
                        'args': args,
                    }, is_best, args.output_dir)

            if epoch + 1 == args.epochs:
                print("=> saving last checkpoint")
                utils.save_on_master({
                    'epoch': 'last',
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_acc1': best_acc1,
                     'best_epoch': best_epoch}
        
        if args.validate_dataset_name_2 and args.validate_dataset_prompt_2:
            log_stats.update({f'test_2_{k}': v for k, v in val_stats_2.items()})

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats, step=wandb_step)
                wandb_step += 1
                # wandb.watch(model)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, lr_schedule_low, args):
    global wandb_step
    
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    utils.get_model(model).open_clip_model.eval()  # set the openCLIP component to eval mode!!

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            #param_group['lr'] = lr_schedule[it]
            if not param_group['block']: # non-block groups
                param_group['lr'] = lr_schedule[min(it, len(lr_schedule) - 1)]
            else:                 # block groups
                param_group['lr'] = lr_schedule_low[min(it, len(lr_schedule_low) - 1)]

        #print(f"input length: {len(inputs)}")
        pc = inputs[2]
        texts = inputs[1]
        image = inputs[3]
        
        #print(f"pc: {pc.shape}, texts: {texts.shape}, image: {image.shape}")
        inputs = [pc, texts, image]

        # Move tensors to device
        if isinstance(inputs[0], Mapping): # pc: Sonata-style Point object
            point = inputs[0]
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].cuda(args.gpu, non_blocking=True)
                    
            inputs[0] = point      
            inputs[1:] = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs[1:]]
        else:
            inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        # compute output
        with amp.autocast(enabled=not args.disable_amp, dtype=torch.bfloat16):
            outputs = model(*inputs)
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue
        
        # Gradient clipping
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]
        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(),
                        'logit': logit_scale},
                        step=wandb_step)
                wandb_step += 1
            progress.display(optim_iter)
            
    # After the main training loop for the epoch
    # Check if there are pending gradients that haven't been applied
    # This happens if len(train_loader) is not a multiple of args.update_freq
    if len(train_loader) > 0 and (len(train_loader) % args.update_freq != 0) and args.update_freq > 1:
        print(f"Epoch [{epoch}]: Performing final optimizer step for remaining {len(train_loader) % args.update_freq} accumulated iterations.")
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True) # Crucially, clear gradients after the step

    progress.synchronize()
    
    return_stats = {k: v.avg for k, v in metrics.items() if k != 'nuscenes_ratio'}
    return_stats['lr'] = optimizer.param_groups[0]['lr'] # Final LR for non-block
    return_stats['lr_block'] = optimizer.param_groups[2]['lr'] # Final LR for block
    return_stats['logit_scale'] = logit_scale

    if 'objects_joint' in args.pretrain_dataset_name:
        return_stats['nuscenes_ratio'] = train_loader.batch_sampler.nuscenes_in_batch / train_loader.batch_sampler.batch_size

    return return_stats
    #return {**{k: v.avg for k, v in metrics.items()},
    #        'lr': optimizer.param_groups[0]['lr'],
    #        'lr_block': optimizer.param_groups[2]['lr'], # LR for block params
    #        'logit_scale': logit_scale,
    #        'nuscenes_ratio': train_loader.sampler.nuscenes_ratio if args.pretrain_dataset_name == 'objects_joint'}


def test_zeroshot_3d_core(test_loader, dataset_name, templates_name, model, tokenizer, args=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    #dataset_name = test_loader.dataset.__class__.__name__.lower()

    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')
    with open(os.path.join("./data", 'templates.json')) as f:
        #templates = json.load(f)[args.validate_dataset_prompt]
        templates = json.load(f)[templates_name]

    if 'objaverse' in dataset_name:
        labels = test_loader.dataset.lvis_metadata['all_keys']
    else:
        with open(os.path.join("./data", 'labels.json')) as f:
            labels = json.load(f)[dataset_name]

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        for i, (pc, target, target_name) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.cuda(args.gpu, non_blocking=True)
                
            target = target.cuda(args.gpu, non_blocking=True)

            # encode pc
            pc_features = utils.get_model(model).encode_pc(pc)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_pc = pc_features @ text_features.t()

            # measure accuracy and record loss
            (acc1, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 5))
            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            
            top1.update(acc1.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                    
                    """
                    import open3d as o3d
                    
                    # Visualize correctly classified point clouds with Open3D
                    
                    point_cloud = pc[idx].cpu().numpy()
                    valid_mask = ~(np.isinf(point_cloud).any(axis=1) | np.isnan(point_cloud).any(axis=1))
                    point_cloud = point_cloud[valid_mask]

                    
                    # Get predicted class name
                    predicted_idx = logits_per_pc[idx].argmax().item()
                    predicted_class = labels[predicted_idx]
                    
                    print(f"Correct prediction! True class: {name}, Predicted class: {predicted_class}")
                    
                     # Visualize with Open3D
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
                    
                    colors = np.tile([0.7, 0.7, 0.7], (len(point_cloud), 1))
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # Visualize point cloud with Open3D
                    print("Displaying point cloud with Open3D...")
                    
                    o3d.visualization.draw_geometries([pcd], 
                                                    window_name=f"Point Cloud: {name}",
                                                    width=800, 
                                                    height=600,
                                                    point_show_normal=False)
                    """
                    
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]
    
        # Print average class-wise accuracies
        avg_top1 = sum(top1_accuracy_per_class.values()) / len(top1_accuracy_per_class)
        avg_top5 = sum(top5_accuracy_per_class.values()) / len(top5_accuracy_per_class)
        print(f"Average class-wise Acc@1: {avg_top1:.4f}")
        print(f"Average class-wise Acc@5: {avg_top5:.4f}")

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        print(','.join(top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        print(','.join([str(value) for value in top5_accuracy_per_class.values()]))

    progress.synchronize()
    print('0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    if torch.cuda.is_available(): # clear cache to prevent OOM when running more than 1 validations back-to-back
        torch.cuda.empty_cache()
        
    return {'acc1': top1.avg, 'acc5': top5.avg}

def test_zeroshot_3d(args):
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    try:
        old_args = ckpt['args']
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> creating model: {}".format(old_args.model))
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> creating model: {}".format(args.model))
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))

    tokenizer = SimpleTokenizer()

    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    results = test_zeroshot_3d_core(test_loader, model, tokenizer, args)

    return results

def test_zeroshot_3d_ulip2(args):
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("=> creating model: {}".format(args.model))

    model, tokenizer, _, _ = getattr(models, args.model)(args=args)
    model.cuda()
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded pretrained checkpoint '{}'".format(args.test_ckpt_addr))

    #tokenizer = SimpleTokenizer()

    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False,
        collate_fn=customized_collate_fn,
        #multiprocessing_context=torch.multiprocessing.get_context("forkserver"),
        persistent_workers=True if args.workers > 0 else False
        )
    
    if args.test_repeat == 1:
        # Single evaluation (original behavior)
        results = test_zeroshot_3d_core(test_loader, args.validate_dataset_name, args.validate_dataset_prompt, model, tokenizer, args)
        return results
    else:
        # Multiple evaluations for statistical analysis
        print(f"=> Running evaluation {args.test_repeat} times for statistical analysis")
        
        acc1_results = []
        acc5_results = []
        
        for run in range(args.test_repeat):
            print(f"=> Evaluation run {run + 1}/{args.test_repeat}")
            
            # Create new dataloader with different seed for shuffling (if any randomness exists)
            # Note: We set shuffle=False above, but there might be other sources of randomness
            torch.manual_seed(args.seed + run)
            np.random.seed(args.seed + run)
            
            results = test_zeroshot_3d_core(test_loader, args.validate_dataset_name, args.validate_dataset_prompt, model, tokenizer, args)
            
            acc1_results.append(results['acc1'])
            acc5_results.append(results['acc5'])
            
            print(f"Run {run + 1}: Acc@1 = {results['acc1']:.3f}, Acc@5 = {results['acc5']:.3f}")
        
        # Calculate statistics
        acc1_mean = np.mean(acc1_results)
        acc1_std = np.std(acc1_results, ddof=1) if len(acc1_results) > 1 else 0.0
        
        acc5_mean = np.mean(acc5_results)
        acc5_std = np.std(acc5_results, ddof=1) if len(acc5_results) > 1 else 0.0
        
        print("\n" + "="*50)
        print("STATISTICAL SUMMARY:")
        print(f"Acc@1: {acc1_mean:.3f} ± {acc1_std:.3f} (mean ± std)")
        print(f"Acc@5: {acc5_mean:.3f} ± {acc5_std:.3f} (mean ± std)")
        print(f"Number of runs: {args.test_repeat}")
        print("="*50)
        
        # Return comprehensive results
        return {
            'acc1': acc1_mean,
            'acc5': acc5_mean,
            'acc1_std': acc1_std,
            'acc5_std': acc5_std,
            'acc1_all_runs': acc1_results,
            'acc5_all_runs': acc5_results,
            'num_runs': args.test_repeat
        }


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
