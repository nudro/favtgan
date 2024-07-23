import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from lpips_pytorch import LPIPS, lpips
import cv2
from torch.distributed import Backend
import antialiased_cnns
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import kornia
from torch.utils.data.sampler import RandomSampler
from datasets_augs import *  # temperature without patches

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--patch_height", type=int, default=128, help="size of patch height")
parser.add_argument("--patch_width", type=int, default=128, help="size of patch width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
parser.add_argument("--gpu_num", type=int, default=0, help="as of 1/8/22 never set this to 2")
parser.add_argument("--out_file", type=str, default="out", help="name of output log files")
parser.add_argument("--experiment", type=str, default="none", help="experiment name")
parser.add_argument("--local_rank", type=int, default=1, help="number GPUs")
opt = parser.parse_args()

"""
Experimental Temperature-Vector Guided Version

Architecture: 
    SmallUNET
    1 Discriminators, 1 Generators
    BLURPOOL
    Spectral Norm
    BCEwithLogits/Relativistic Loss
    
Losses:
    LPIPS + Triplet Loss for Temp
    No Patches as inputs - Only Global Image
    Also no patches for triplet loss.
    Learning only on global image. 

Dataset: 
    Uses augmentations through random sampling two datasets 
    for Iris and Devcom, only 
    Patches do not work well for these datasets
    For Eurecom, patches seem to work well. 

You can use this for a single dataset, like training only on Eurecom. 

Training:
    > Loads in only A,B 
    > AMP
    > HalfTensors
    > BlurPool() up and down sampling on G and 2D's
    => say in my paper quote, "anti-aliasing to improve shiftequivariance of deep networks"
    > Randomized patches for TripletLoss for negative patches
    > batch size = 256

"""

os.makedirs("/home/local/AD/cordun1/experiments/faPVTgan/images/%s" % opt.experiment, exist_ok=True)
os.makedirs("/home/local/AD/cordun1/experiments/faPVTgan/saved_models/%s" % opt.experiment, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(opt.gpu_num) 

# Set fixed random number seed
torch.manual_seed(42)

#################
# Loss functions
#################

criterion_GAN = torch.nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class.
criterion_lpips = LPIPS(
    net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)
# scaling param for temp loss since value will be very small
lambda_t = 10
criterion_temp = nn.TripletMarginLoss(margin=1.0, p=2)


################
# Discr. Patches
#################

# Calculate output of image discriminator (PatchGAN)
patch_for_g = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
print("discriminator patch for global:", patch_for_g) # (1, 16, 16) = N X N

##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 1, 1, bias=False)] # originally stride = 2
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(antialiased_cnns.BlurPool(out_size, stride=2)) #downsample stride2
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
                nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                antialiased_cnns.BlurPool(out_size, stride=1), #upsampling with stride=1 is just blurring no pooling
                nn.InstanceNorm2d(out_size),
                nn.ReLU(inplace=True),
            ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

############
# G1
############

class GeneratorUNet1(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet1, self).__init__()
        channels, self.h, self.w = img_shape
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, normalize = False)
        self.down6 = UNetDown(512, 512)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        with autocast():
            d1 = self.down1(x)
            d2 = self.down2(d1)  
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            u1 = self.up1(d6, d5)
            u2 = self.up2(u1, d4)
            u3 = self.up3(u2, d3)
            u4 = self.up4(u3, d2)
            u5 = self.up5(u4, d1)
            output = self.final(u5).type(HalfTensor)
        return output

    
##############################
#        Discriminator1
# Uses Spectral Norm instead of Instance Norm
##############################

class Discriminator1(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator1, self).__init__()
        channels, self.h, self.w = img_shape

        def discriminator_block(in_filters, out_filters):
            layers = [torch.nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_filters, out_filters, 4, stride=1, padding=1))] # changed to stride=1 instead of 2

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(antialiased_cnns.BlurPool(out_filters, stride=2)) #blurpool downsample stride=2
            
            return layers

        self.model = nn.Sequential(
            *discriminator_block((channels * 2), 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            d_in = img_input
            output = self.model(d_in).type(HalfTensor)
        return output


##############################
#       Utils
##############################

# for passing to vectorizer for fake_B temperature calculation
T = np.linspace(24, 38, num=256) # 0 - 255 indices of temperatures in Celsius
d = dict(enumerate((T).flatten(), 0)) # dictionary like {0: 24.0, 1: 24.054901960784314, etc.}

def vectorize_temps(fake_B):
    TFB = []
    for t in range(0, opt.batch_size):
        b = transforms.ToPILImage()(fake_B[t, :, :, :]).convert("RGB")
        fake_vectorizer = TempVector_PyTorch(b, d) # calls datasets_temp .detach().cpu()
        tfb = torch.Tensor(fake_vectorizer.make_pixel_vectors()).cuda()
        TFB.append(tfb)
    TFB_tensor = torch.cat(TFB).reshape(opt.batch_size, 1, opt.img_height, opt.img_width)     
    return TFB_tensor
 
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

        
def concat_output(X, idx):
    #X is the datatype loaded
    main = Variable(batch1[X].type(HalfTensor)) #HalfTensor
    aug = Variable(batch2[X].type(HalfTensor))
    out = torch.cat((main, aug), dim=0)
    out = out[idx].view(out.size())
    return out

        
def sample_images(batches_done):
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs["A"]).cuda()
    real_B = Variable(imgs["B"]).cuda()
    fake_B = generator1(real_A).cuda()
    
    # GLOBAL
    img_sample_global = torch.cat((real_A.data, fake_B.data, fake_B2, real_B.data), -2)
    save_image(img_sample_global, "images/%s/%s_g.png" % (opt.experiment, batches_done), nrow=5, normalize=True)


##############################
#       Initialize
##############################

input_shape_global = (opt.channels, opt.img_height, opt.img_width)

generator1 = GeneratorUNet1(input_shape_global)
discriminator1 = Discriminator1(input_shape_global)

generator1 = generator1.cuda()
discriminator1 = discriminator1.cuda()

criterion_GAN.cuda()
criterion_lpips.cuda()
criterion_temp.cuda()

################    
# nn.DataParallel
################

generator1 = torch.nn.DataParallel(generator1, device_ids=[1, 2])
discriminator1 = torch.nn.DataParallel(discriminator1, device_ids=[1, 2])

if opt.epoch != 0:
    # Load pretrained models
    generator1.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/faPVTgan/saved_models/%s/generator1_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator1.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/faPVTgan/saved_models/%s/discriminator1_%d.pth" % (opt.experiment, opt.epoch)))

else:
    # Initialize weights
    generator1.apply(weights_init_normal)
    discriminator1.apply(weights_init_normal)


################    
# Optimizers
################
optimizer_D = torch.optim.Adam(discriminator1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


##############################
# Transforms and Dataloaders
##############################

# Only for the MainDataset() no augmentations
# Remember resize is done in the MainDataset()
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


main_dataloader = DataLoader(
    MainDataset(root = "/home/local/AD/cordun1/experiments/data/iris_v3_pairs",
                 transforms_=transforms_,
                mode="train"), 
    batch_size=opt.batch_size//2, # for exa: if batch_size 64 for both datasets, 32 to eur and 32 to dev, making it 1:1
    shuffle=True,
    num_workers=8,
    drop_last=True,
)


random_sampler = RandomSampler(AugDataset(root = "/home/local/AD/cordun1/experiments/data/iris_v3_pairs",
                 transforms_= None,
                mode="train"))

aug_dataloader = DataLoader(
    AugDataset(root = "/home/local/AD/cordun1/experiments/data/iris_v3_pairs",
                 transforms_= None,
                mode="train"), 
    batch_size=opt.batch_size//2,
    sampler=random_sampler,
    num_workers=8,
    drop_last=True,
)

#====TEST DATALOADER====

test_dataloader = DataLoader(
    TestDataset(root = "/home/local/AD/cordun1/experiments/data/iris_v3_pairs",
        transforms_=transforms_,
        mode="test"),
    batch_size=1*3, #all 3 GPUs
    shuffle=True,
    num_workers=8,
)

# Tensor type - only use HalfTensor in this AMP script
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

##############################
#       Training
##############################

prev_time = time.time()

f = open('/home/local/AD/cordun1/experiments/faPVTgan/LOGS/{}.txt'.format(opt.out_file), 'a+')

# Try AMP
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch1 in enumerate(main_dataloader): 
        try:
            aug_iterator = iter(aug_dataloader)
            batch2 = next(aug_iterator) # batch2 is augmentations
        except StopIteration:
            main_iterator = iter(main_dataloader)
            batch2 = next(main_iterator)

        # Let real_A be run without using a function b/c I need the indices (idx)
        real_A_main = Variable(batch1["A"].type(HalfTensor))
        real_A_aug = Variable(batch2["A"].type(HalfTensor))
        real_A = torch.cat((real_A_main, real_A_aug), dim=0).type(HalfTensor)
        
        # randomizes tensors by the batch index
        # so that aug doesn't always follow main
        # keep idx throught the iteration so that the pairs align
        idx = torch.randperm(real_A.shape[0]) # change by the 0th (batches)
        real_A = real_A[idx].view(real_A.size())
        
        #concat_output fxn returns HalfTensors
        real_B = concat_output("B", idx)
        
        #concat_output fxn returns HalfTensors
        TB = concat_output("T_B", idx)

        # Adversarial ground truths: global image
        valid_ones = Variable(HalfTensor(np.ones((real_A.size(0), *patch_for_g))), requires_grad=False)
        valid = valid_ones.fill_(0.9)
        fake = Variable(HalfTensor(np.zeros((real_A.size(0), *patch_for_g))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        print("+ + + optimizer_G.zero_grad() + + +")
        optimizer_G.zero_grad()

        with autocast(): 
            fake_B = generator1(real_A)
            #print("fake_B size:", fake_B.size())
            
            # Adversarial - relativistic loss
            pred_fake = discriminator1(fake_B, real_A)
            real_pred = discriminator1(real_B, real_A)
            loss_GAN = criterion_GAN(pred_fake - real_pred.detach(), valid)
             
            # TO DO - WRAP THIS UP IN A FUNCTION    
            # Temperature loss using Triplet Loss
            TFB_ = vectorize_temps(fake_B) # passes tensor
            
            # data augmented B temps, serves as negatives
            # hence, this is another augmentation dealing with color, even on the geometric augs
            transform_jit = transforms.ColorJitter(brightness=0.5, contrast=0.75, saturation=1.5, hue=0.5)
            B_tf = transform_jit(real_B)
            TBTF = vectorize_temps(B_tf)
            
            TB = TB.reshape(TB.size(0), 1, TB.size(1), TB.size(2))
            
            #(TFB_ anchor, TB ground truth, TBTF negatives)
            loss_temp_g = criterion_temp(TFB_, TB, TBTF)*lambda_t
            
            # LPIPS loss - perceptual similarity for fake_B
            loss_pix = criterion_lpips(fake_B, real_B)

            # Total Generator Loss
            loss_G = loss_GAN + loss_temp + loss_pix

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        print("+ + + optimizer_G.step() + + +")

        # ----------------------
        #  Train Discriminator 1
        # ----------------------

        print("+ + + optimizer_D.zero_grad() + + +")
        optimizer_D.zero_grad()
        
        with autocast():
            # real
            pred_real = discriminator1(real_B, real_A)
            #fake_B 
            pred_fake_B = discriminator1(fake_B.detach(), real_A)

            #adv loss
            loss_real = criterion_GAN(pred_real - pred_fake_B, valid) 
            loss_fake = criterion_GAN(pred_fake_B - pred_real, fake) 
            loss_D = 0.5*(loss_real + loss_fake)

        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        print("+ + + optimizer_D.step() + + +")
        
        # one scalar update for all scaler
        # instead of one for each model
        scaler.update()

        # --------------
        #  Log Progress
        # --------------

        batches_done = epoch * len(main_dataloader) + i  
        batches_left = opt.n_epochs * len(main_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r |Experiment: %s| [Epoch %d/%d] [Batch %d/%d] [D_real %f | D_fake %f | D_loss %f] [G_loss %f | GAN %f | pix %f | temp %f ] ETA: %s"
            % (
                opt.experiment, 
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(main_dataloader), #%d
                loss_real.item(),
                loss_fake.item(),
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                loss_GAN.item(),
                loss_pix.item(),
                loss_temp.item(),
                time_left, #%s
            )
        )

        f.write(
            "\r |Experiment: %s| [Epoch %d/%d] [Batch %d/%d] [D_real %f | D_fake %f | D_loss %f] [G_loss %f | GAN %f | pix %f | temp %f ] ETA: %s"
            % (
                opt.experiment, 
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(main_dataloader), #%d
                loss_real.item(),
                loss_fake.item(),
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                loss_GAN.item(),
                loss_pix.item(),
                loss_temp.item(),
                time_left, #%s
            )
        )
         # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

                
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator1.state_dict(), "/home/local/AD/cordun1/experiments/faPVTgan/saved_models/%s/generator1_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator1.state_dict(), "/home/local/AD/cordun1/experiments/faPVTgan/saved_models/%s/discriminator1_%d.pth" % (opt.experiment, epoch))
        

f.close()
