import hydra
import torch
from utils.util import AverageMeter, convert_layers
from utils.dataset import BlenderLaptopAuxDataset
import numpy as np
import logging
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import segmentation
import os


@hydra.main(config_path='./config', config_name='laptop_aux')
def main(cfg):
    logger = logging.getLogger(__name__)

    # load name
    name_path = hydra.utils.to_absolute_path('data/shapenet_names/{}.txt'.format(cfg.category))
    if os.path.exists(name_path):
        shapenames = open(name_path).read().splitlines()

    ds = BlenderLaptopAuxDataset(cfg, shapenames)
    df = torch.utils.data.DataLoader(ds, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=10)
    
    segmenter = segmentation.fcn_resnet50(num_classes=2).cuda()
    segmenter = convert_layers(segmenter, nn.BatchNorm2d, nn.InstanceNorm2d).cuda()

    opt = optim.Adam([*segmenter.parameters()], lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
    ce = nn.CrossEntropyLoss()
    
    logger.info('Train')
    for epoch in range(cfg.max_epoch):
        n = 0
        
        loss_meter = AverageMeter()
        segmenter.train()
        with tqdm(df) as t:
            for rgb, label in t:
                rgb, label = rgb.cuda(), label.cuda()
                
                opt.zero_grad()
                
                feat = segmenter(rgb.permute(0, 3, 1, 2))['out']  # B x 2 x H x W
                
                loss = ce(feat, label)
                loss.backward(retain_graph=False)
                
                opt.step()
                
                loss_meter.update(loss.item())
                    
                n += 1
                t.set_postfix(loss=loss_meter.avg)
            
        torch.save(segmenter.state_dict(), f'segmenter_current.pth')
        logger.info('loss: {:.4f}'.format(loss_meter.avg))

if __name__ == '__main__':
    main()