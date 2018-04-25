import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import time
import os
import shutil
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from dataset import ObfuscatedDatasetLoader
from models.resnet_subpixel_cnn_variant import ResnetSubPixelCNNVariant
from models.discriminator_cnn import DiscriminatorCNN
from scripts.metrics import calc_psnr
from loss import create_loss_model
from torchvision import models
from math import log10

def to_var(x):
    if torch.cuda.is_available():#utilize the gpu of the pc
        x = x.cuda()
    return Variable(x)


def train(train_loader, modelG, modelD, loss_type, optimizerG, optimizerD, epoch, vgg_loss, model_name):
    """ Train the model for one epoch """

    batch_time_meter = AverageMeter()
    losses_meter = AverageMeter()
    psnr_meter = AverageMeter()
    # setup the loss function (MSE/BCE)
    loss_fn = nn.MSELoss().cuda()
    loss_fn2 = nn.BCELoss().cuda()

    # set the model to train mode
    modelD.train()
    modelG.train()

    start = time.time()

    for iteration, batch in enumerate(train_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        # ==================================================================
        # TRAINING THE DISCRIMINATIVE MODEL
        # ==================================================================
        input = input.cuda()
        target = target.cuda()
        real_labels = to_var(torch.ones(batch_size))
        fake_labels = to_var(torch.zeros(batch_size))

        # Feed Discriminator GT images, and labels = REAL
        outputs = modelD(target)
        d_loss_real = loss_fn2(outputs, real_labels)
        real_score = outputs

        # Feed Discriminator Generated images, and labels = FAKE
        fake_images = modelG(input)
        outputs = modelD(fake_images)
        d_loss_fake = loss_fn2(outputs, fake_labels)
        fake_score = outputs

        # Update Gradients for Discriminator
        lossD = d_loss_real + d_loss_fake
        modelD.zero_grad()
        lossD.backward()
        optimizerD.step()
        # ==================================================================
        # TRAINING THE GENERATIVE MODEL
        # ==================================================================

        fake_images = modelG(input)
        outputs2 = modelD(fake_images)
        if loss_type == "perceptual":
            vgg_loss_output = vgg_loss(fake_images)
            vgg_loss_target = vgg_loss(target)
            lossG = (loss_fn(vgg_loss_output, vgg_loss_target)*.3) + (loss_fn2(outputs2,fake_labels)*.7)
        else:
            lossG = (loss_fn(fake_images,target)*.3) + (loss_fn2(outputs2, fake_labels)*.7)

        # Backprop + Optimize
        modelD.zero_grad()
        modelG.zero_grad()
        lossG.backward()
        optimizerG.step()
        # ==================================================================
        # UPDATING STATISTICS
        # ==================================================================

        # measure psnr and loss
        mse = loss_fn(fake_images, target)
        psnr = 10 * log10(1 / mse.data[0])
        psnr_meter.update(psnr, input.size(0))
        losses_meter.update(lossG.data[0], input.size(0))

        # measure the time it takes to train for one epoch
        batch_time_meter.update(time.time() - start)
        start = time.time()

        # ==================================================================
        # PRINTING STATISTICS
        # ==================================================================

        if iteration % 500 == 0:
            print('Epoch [%d], Step[%d/%d], d_loss: %.4f, ''g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f' % (epoch, iteration + 1, len(train_loader), lossD.data[0], lossG.data[0], real_score.data.mean(), fake_score.data.mean()))

            new_output_dir = "./images_from_runs/{0}/train/".format(model_name)

            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir)

            model_output_image = fake_images.data.float()
            model_input_image = input.data.float()
            model_target_image = target.data.float()

            if opt.save_img:
                filename = new_output_dir + "{0}_epoch_{1}_iter.jpg".format(epoch, iteration)
                save_image(input=model_input_image, output=model_output_image, target=model_target_image,
                           filename=filename)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PSNR {psnr.val:.3f} ({psnr.avg:.3f})'.format(epoch, iteration, len(train_loader), batch_time=batch_time_meter,loss=losses_meter, psnr=psnr_meter))

    # log value to tensorboard or visdom
    if opt.tensorboard:
        writer.add_scalar("PSNR/train ", psnr_meter.avg, epoch)
        writer.add_scalar("Loss/train", losses_meter.avg, epoch)


def validate(val_loader, modelG, modelD, loss_type, epoch, vgg_loss, model_name):
    """ Validate the model on the validation set """
    batch_time_meter = AverageMeter()
    losses_meterG = AverageMeter()
    losses_meterD = AverageMeter()
    psnr_meter = AverageMeter()
    loss_fn = nn.MSELoss().cuda()
    loss_fn2 = nn.BCELoss().cuda()

    # switch to eval mode
    modelG.eval()
    modelD.eval()
    start = time.time()

    for iteration, batch in enumerate(val_loader, start=1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        # ==================================================================
        # EVALUATING THE DISCRIMINATIVE MODEL
        # ==================================================================
        input = input.cuda()
        target = target.cuda()
        real_labels = to_var(torch.ones(input.size(0)))
        fake_labels = to_var(torch.zeros(input.size(0)))

        # Feed Discriminator GT images, and labels = REAL
        outputs = modelD(target)
        d_loss_real = loss_fn2(outputs, real_labels)
        real_score = outputs

        #  Feed Discriminator Generated images, and labels = FAKE
        fake_images = modelG(input)
        outputs = modelD(fake_images)
        d_loss_fake = loss_fn2(outputs, fake_labels)
        fake_score = outputs
        lossD = d_loss_real + d_loss_fake
        # ==================================================================
        # EVALUATING THE GENERATIVE MODEL
        # ==================================================================

        fake_images = modelG(input)
        outputs2 = modelD(fake_images)
        if loss_type == "perceptual":
            vgg_loss_output = vgg_loss(fake_images)
            vgg_loss_target = vgg_loss(target)
            lossG = (loss_fn(vgg_loss_output, vgg_loss_target)*.3) + (loss_fn2(outputs2,fake_labels)*.7)
        else:
            lossG = (loss_fn(fake_images,target)*.3) + (loss_fn2(outputs2, fake_labels)*.7)
        # ==================================================================
        # UPDATING STATISTICS
        # ==================================================================

        # compute the psnr and loss on the validation set

        # measure psnr and loss
        mse = loss_fn(fake_images, target)
        psnr = 10 * log10(1 / mse.data[0])
        psnr_meter.update(psnr, input.size(0))
        losses_meterG.update(lossG.data[0], input.size(0))
        losses_meterD.update(lossD.data[0], input.size(0))


        # measure time
        batch_time_meter.update(time.time() - start)
        start = time.time()
        # ==================================================================
        # PRINTING STATISTICS
        # ==================================================================

        if iteration % 100 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, ''g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f' % (epoch, 200, iteration + 1, 600, lossD.data[0], lossG.data[0], real_score.data.mean(),fake_score.data.mean()))

            new_output_dir = "./images_from_runs/{0}/val/".format(model_name)

            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir)

            model_output_image = fake_images.data.float()
            model_input_image = input.data.float()
            model_target_image = target.data.float()

            if opt.save_img:
                filename = new_output_dir + "{0}_epoch_{1}_iter.jpg".format(epoch, iteration)
                save_image(input=model_input_image, output=model_output_image, target=model_target_image,
                           filename=filename)

            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PSNR {psnr.val:.3f} ({psnr.avg:.3f})'.format(iteration, len(val_loader), batch_time=batch_time_meter, loss=losses_meter,psnr=psnr_meter))

    print("AVG PSNR after epoch {0}: {1}".format(epoch, psnr_meter.avg))

    if opt.tensorboard:
        writer.add_scalar("PSNR/val", psnr_meter.avg, epoch)
        writer.add_scalar("Loss/val", losses_meter.avg, epoch)

    return losses_meterG.avg, losses_meterD.avg, psnr_meter.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def save_image(input, output, target, filename):
    """ Save the input, output, target image during training """
    all_images = torch.cat((input, output, target))
    vutils.save_image(all_images, filename=filename, normalize=True)


def save_checkpoint(name, state, is_best, filename='checkpoint.pth.tar'):
    """Saves model checkpoint to disk"""
    directory = "saved_models/%s/" % (name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'saved_models/%s/' % (name) + 'model_best.pth.tar')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Reconstruction using CNNs')
    parser.add_argument("--model", type=str, default="ThreeLayerCNNBasline",
                        help="type of model to use for facial reconstruction")
    parser.add_argument("--method", type=str, default="blurred", help="type of obfuscation method to use")
    parser.add_argument("--size", type=int, help="size of the obfuscation method applied to images")
    parser.add_argument('--grayscale', action="store_true", help="use grayscale images?")
    parser.add_argument("--loss", type=str, default="mse", help="type of loss function to use (eg. mse, perceptual)")
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="weight decay applied to the optimizer")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--tensorboard', action="store_true", help="use tensorboard for visualization?")
    parser.add_argument('--save_img', action="store_true", help="save the output images when training the model")
    parser.add_argument('--pre_pochs', type=int, default=5, help="number of epochs for pre-training Generator")
    global opt, writer, best_avg_psnr
    opt = parser.parse_args()

    # setup the tensorboard
    writer = SummaryWriter("./runs/")

    best_avg_psnrG = 0
    best_avg_psnrD = 0

    # get the arguments from argparse
    num_epochs = opt.epochs
    lr = opt.lr
    method = opt.method
    size = opt.size
    batch_size = opt.batch_size
    use_cuda = opt.cuda
    loss_type = opt.loss
    num_workers = opt.threads
    weight_decay = opt.weight_decay
    grayscale = opt.grayscale
    pre_epochs = opt.pre_pochs

    main_hyperparametersG = "{0}_method={1}_size={2}_loss={3}_lr={4}_epochs={5}_batch_size={6}".format(opt.model, opt.method, opt.size, opt.loss, opt.lr, opt.epochs, opt.batch_size)
    print("Hyperparameters Gene: ", main_hyperparametersG)

    main_hyperparametersD = "{0}_method={1}_size={2}_loss={3}_lr={4}_epochs={5}_batch_size={6}".format("Discriminative", opt.method, opt.size, opt.loss, opt.lr, opt.epochs, opt.batch_size)
    print("Hyperparameters Disc: ", main_hyperparametersD)

    if grayscale:
        image_color = "grayscale"
    else:
        image_color = "rgb"

    #################
    # Normalization #
    #################
    train_mean = np.array([149.59638197, 114.21029544, 93.41318133])
    train_std = np.array([52.54902009, 44.34252746, 42.88273568])
    normalize = transforms.Normalize(mean=[mean / 255.0 for mean in train_mean],
                                     std=[std / 255.0 for std in train_std])

    transform_normalize = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # get the training data
    train_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped_{}/".format(image_color), method, size,
                                         grayscale=False, data_type="train", transform=transform_normalize)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # get the validation set
    val_dset = ObfuscatedDatasetLoader("./data/lfw_preprocessed/cropped_{}/".format(image_color), method, size,
                                       grayscale=False, data_type="val", transform=transform_normalize)
    val_loader = DataLoader(val_dset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    # ============================
    # GENERATIVE MODEL
    # ============================

    # get the model Generative
    modelG = ResnetSubPixelCNNVariant()

    # set the optimizer Generative
    optimizerG = optim.Adam(modelG.parameters(), lr=lr, weight_decay=weight_decay)

    # set the scheduler Generative
    schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, "min", patience=5)

    if use_cuda:
        modelG = modelG.cuda()

    # ============================
    # DISCRIMINATIVE MODEL
    # ============================

    # get the model Discriminative
    modelD = DiscriminatorCNN()

    # set the optimizer Discriminative
    optimizerD = optim.Adam(modelD.parameters(), lr=lr, weight_decay=weight_decay)

    # set the scheduler Discriminative
    schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, "min", patience=5)

    if use_cuda:
        modelD = modelD.cuda()

    # ==============================
    # VGG MODEL for PERCEPTUAL LOSS
    # ==============================
    vgg16 = models.vgg16(pretrained=True).features
    vgg16.cuda()
    vgg_loss = create_loss_model(vgg16, 8, use_cuda=True)

    for param in vgg_loss.parameters():
        param.requires_grad = False

    batch_time_meter = AverageMeter()
    losses_meter = AverageMeter()
    psnr_meter = AverageMeter()
    # setup the loss function (MSE/BCE)
    loss_fn = nn.MSELoss().cuda()
    loss_fn2 = nn.BCELoss().cuda()

    # ==============================
    # PRETRAINING GENERATIVE MODEL
    # ==============================
    for i in range(pre_epochs):
        # set the model to train mode
        modelG.train()
        start = time.time()
        for iteration, batch in enumerate(train_loader, 1):
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            # ==================================================================
            # TRAINING THE DISCRIMINATIVE MODEL
            # ==================================================================
            input = input.cuda()
            target = target.cuda()
            real_labels = to_var(torch.ones(batch_size))
            fake_labels = to_var(torch.zeros(batch_size))

            # ==================================================================
            # TRAINING THE GENERATIVE MODEL
            # ==================================================================

            fake_images = modelG(input)
            if loss_type == "perceptual":
                vgg_loss_output = vgg_loss(fake_images)
                vgg_loss_target = vgg_loss(target)
                lossG = (loss_fn(vgg_loss_output, vgg_loss_target))
            else:
                lossG = (loss_fn(fake_images, target))

            # Backprop + Optimize
            modelG.zero_grad()
            lossG.backward()
            optimizerG.step()
            # ==================================================================
            # UPDATING STATISTICS
            # ==================================================================

            # measure psnr and loss
            mse = loss_fn(fake_images, target)
            psnr = calc_psnr(mse.data[0], input.size(0))
            psnr_meter.update(psnr)
            losses_meter.update(lossG.data[0], input.size(0))

            # measure the time it takes to train for one epoch
            batch_time_meter.update(time.time() - start)
            start = time.time()

            # ==================================================================
            # PRINTING STATISTICS
            # ==================================================================
            if iteration % 500 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'PSNR {psnr.val:.3f} ({psnr.avg:.3f})'.format(i, iteration, len(train_loader),
                                                                    batch_time=batch_time_meter, loss=losses_meter,
                                                                    psnr=psnr_meter))
    # ==============================
    # DONE TRAINING GENERATIVE MODEL
    # ==============================
    # =============================
    # RUNNING FORLOOP OVER EPOCHS
    # =============================
    for epoch in range(num_epochs):
        # trains the model for one epoch
        train(train_loader, modelG, modelD, loss_type, optimizerG, optimizerD, epoch, vgg_loss, model_name=main_hyperparametersG)

        # ==========================================================
        # evaluate on the validation set Generative
        # ==========================================================
        val_lossG, val_lossD, val_psnr_avgG, = validate(val_loader, modelG, modelD, loss_type, epoch, vgg_loss,
                                          model_name=main_hyperparametersG)

        # adjust the learning rate if val loss stops improving
        schedulerG.step(val_lossG)

        # adjust the learning rate if val loss stops improving
        schedulerD.step(val_lossD)

        # ==========================================================
        # remember the best psnr value and save the checkpoint model
        # ==========================================================
        is_bestG= val_psnr_avgG > best_avg_psnrG
        best_avg_psnrG = max(val_psnr_avgG, best_avg_psnrG)
        save_checkpoint(main_hyperparametersG, {
            'epoch': epoch + 1,
            'state_dict': modelG.state_dict(),
            'best_psnr': best_avg_psnrG,
        }, is_bestG)
    print("Best PSNR on Gene the validation set: {}".format(best_avg_psnrG))