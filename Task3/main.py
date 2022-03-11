import argparse
from asyncore import write
from decimal import MAX_PREC
import math
from re import X
import numpy as np
import cv2 

from tqdm import tqdm
from datetime import datetime
import os
from dataloader import get_cifar10, get_cifar100
from utils      import accuracy, AverageMeter
from random import randrange

from model.wrn  import WideResNet

import torch
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data   import DataLoader, RandomSampler, SequentialSampler

from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

def tensor2img(args, image):

    if args.num_classes == "10":
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2471, 0.2435, 0.2616])
    else:
        mean   = np.array([0.5071, 0.4867, 0.4408])
        std    = np.array([0.2675, 0.2565, 0.2761])

    tf = transforms.Compose([
        transforms.Normalize((-mean/std), (1/std))
    ])

    img = image.detach().cpu()
    img = tf(img).numpy()
    # print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype('uint8')
    return img

def img2tensor(args, image):

    if args.num_classes == "10":
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2471, 0.2435, 0.2616])
    else:
        mean   = np.array([0.5071, 0.4867, 0.4408])
        std    = np.array([0.2675, 0.2565, 0.2761])

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    return tf(image)

def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
 
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
 
    return normalized_cdf
 
def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table
 
def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)
 
    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0,256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0,256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0,256])    
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0,256])    
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0,256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0,256])
 
    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)
 
    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)
 
    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)
 
    # Put the image back together
    image_after_matching = cv2.merge([
        blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)
 
    return image_after_matching

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        args.model_depth = 28
        args.model_width = 2
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        args.model_depth = 28
        args.model_width = 8
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)

    # print(labeled_dataset)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=args.train_batch,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.train_batch*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.train_batch,
        num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width, dropRate=0)
    model       = model.to(device)

    if args.use_ema:
        from ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay, device)

    ############################################################################
    # TODO: SUPPLY your code
    if args.resume:
        writer = SummaryWriter(args.resume+'/log')
        path = args.resume
    else:
        d = datetime.now().strftime('%Y-%m-%d~%H:%M:%S')
        path = './' + args.dataset + '' + str(args.num_labeled) + '' + str(args.threshold) + '_' +  d

        if not os.path.exists(path + '/ckpt'):
            os.makedirs(path + '/ckpt')
            os.makedirs(path + '/log')
        save_tbx_log = path + '/log'
        writer = SummaryWriter(save_tbx_log)
        

    # writer = SummaryWriter('./log')
    # criterion = torch.nn.CrossEntropyLoss()

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wd},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optim
    optim = torch.optim.SGD(
            grouped_parameters,
            lr=0.03,
            momentum=0.9,
            nesterov=True
        )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=1e-2, steps_per_epoch=1024, epochs=10)
    ############################################################################
    
    best_acc = 0
    args.start_epoch = 0

    if args.resume: # give path where the ckpt is stored
        print('=======> RESUMING <=======')

        checkpoint = torch.load(args.resume + '/ckpt/' + args.dataset + '.pth.tar')
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    
    model.zero_grad()
    model.train()
    for epoch in range(args.start_epoch, args.epoch):

        print('-'*30)
        print("epoch: ", epoch)
        print('-'*30)
        train_loss_epoch = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_hist = AverageMeter()

        tbar = tqdm(range(args.iter_per_epoch))

        for i in tbar:
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
            inputs_u_w, inputs_u_s = inputs_u_w.to(device), inputs_u_s.to(device)
            ####################################################################
            # TODO: SUPPLY your code

            
#             print('Train X: ', x_l.shape)
#             print('Train X_ul_w: ', x_ul_w.shape)
#             print('Train X_ul_s: ', x_ul_s.shape)
            
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(device)

            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # zero the parameter gradients
            optim.zero_grad()

            '''
            ----------------------------- fixmatch loss -------------------------------------
            referred from: https://github.com/kekmodel/FixMatch-pytorch
            '''
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.threshold, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            fix_match_loss = Lx + Lu
            '''
            ----------------------------- fixmatch loss -------------------------------------
            '''
            mask_idx = torch.nonzero(mask)[:, 0] # get index of max_probs > args.threshold
            target_select_uw = torch.index_select(targets_u, 0, mask_idx) # select labels of weak aug > args.threshold
            target_unique_uw = torch.unique(target_select_uw)
            # print('target_unique_uw: ', target_unique_uw)
            '''
            ----------------------------- HIST MATCHING -------------------------------------
            '''
            with torch.no_grad():
                if (target_unique_uw.size(0) != 0):
                    select_unlabel_samples = torch.empty(target_unique_uw.size(0), 3, 32, 32).to(device)
                    select_label_samples = torch.empty(target_unique_uw.size(0), 3, 32, 32).to(device)
                    select_label_output = torch.empty(target_unique_uw.size(0)).to(device)
                    
                    # print('unlabel unique: ', unlabel_unique.size())
                
                    for k in range(target_unique_uw.size(0)):
                        idx_unlabel = torch.where(targets_u == target_unique_uw[k])[0]
                        idx_label = torch.where(targets_x == target_unique_uw[k])[0]
                        #print('idx_unlabel: ', idx_unlabel.size())
                        #print('idx_label: ', idx_label.size())
                        if (idx_unlabel.size(0) != 0 and idx_label.size(0) !=0 ):
                            #print('select_uindex: ', idx_unlabel_unique.shape)
                            select_uindex = randrange(0, idx_unlabel.size(0))
                        
                            select_unlabel_samples[k, :, :, :] = torch.index_select(inputs_u_w, 0, torch.as_tensor(idx_unlabel[select_uindex], device=device))
                
                            select_lindex = randrange(0, idx_label.size(0))

                            select_label_samples[k, :, :, :] = torch.index_select(inputs_x, 0, idx_label[select_lindex])
                            select_label_output[k] = torch.index_select(targets_x, 0, idx_label[select_lindex])

                    # print('select_label_samples size: ', select_label_samples.size())
                    # print('select_label_output size: ', select_label_output)
                    # print('select_unlabel_output size: ', select_unlabel_samples.size())

                    hist_matched_tensor = torch.empty(select_label_samples.size(0), 3, 32, 32).to(device)
                    for i in range(select_unlabel_samples.size(0)): #max can args.num_classes
                        # print(i)
                        img_label = tensor2img(args, select_label_samples[i, :, :, :])
                        img_unlabel = tensor2img(args, select_unlabel_samples[i, :, :, :])
                        '---------------------------HIST MATCHING-----------------------------------'
                        # hist matching referred from: https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/
                        
                        image_label_matched = match_histograms(img_label, img_unlabel)
                        # cv2.imwrite('./img_matched.png', image_label_matched)
                        # cv2.imwrite('./img_label.png', img_label)
                        # cv2.imwrite('./img_unlable.png', img_unlabel)
                        '---------------------------HIST MATCHING-----------------------------------'
                        hist_matched_tensor[i, :, :, :] = img2tensor(args, image_label_matched)

                    # print(hist_matched_tensor.size())

            if (target_unique_uw.size(0) != 0):
                hist_matched_pred = model(hist_matched_tensor)
                hist_loss = F.cross_entropy(hist_matched_pred, select_label_output.long())
                try:
                    losses_hist.update(hist_loss.detach().cpu().item())
                except:
                    continue
                final_loss = fix_match_loss + hist_loss
            else:
                final_loss = fix_match_loss
            
            final_loss.backward()

            train_loss_epoch.update(final_loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            

            optim.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)

            model.zero_grad()

            tbar.set_description('Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_hist: {loss_hist:.4f}. '.format(
                loss=train_loss_epoch.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                loss_hist=losses_hist.avg))

        '''
        ---------------------- TESTING ----------------------
        '''

        with torch.no_grad():
            test_loss = AverageMeter()
            test_acc = AverageMeter()
            # total = 0
            if args.use_ema:
                test_model = ema_model.ema
            else:
                test_model = model

            test_loader = tqdm(test_loader)
            for i, (data, target) in enumerate(test_loader):
                test_model.eval()
                data, target = data.to(device), target.to(device)
                pred = test_model(data)
                loss = F.cross_entropy(pred, target)

                test_loss.update(loss.item(), data.shape[0])
                test_acc.update(accuracy(pred, target)[0].item(), data.shape[0])

                test_loader.set_description("Loss: {loss:.4f}. Test Acc: {acc:.4f}. ".format(
                    loss=test_loss.avg,
                    acc=test_acc.avg,
                ))

        writer.add_scalar('Train/Loss', train_loss_epoch.avg, epoch)
        writer.add_scalar('Test/Acc', test_acc.avg, epoch)
        writer.add_scalar('Test/loss', test_loss.avg, epoch)

        if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                ema_model.ema, "module") else ema_model.ema

        if (test_acc.avg > best_acc):
            best_acc = test_acc.avg
            torch.save(
                {
                'best_acc': best_acc,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                }, path + '/ckpt/cifar10.pth.tar')
            
            ####################################################################
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num-labeled', type=int, 
                        default=250, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=5e-4, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train_batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test_batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=2*10, type=int,
                        help='total number of iterations to run') # 512
    parser.add_argument('--iter-per-epoch', default=2, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)