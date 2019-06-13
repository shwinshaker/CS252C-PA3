import torch
from torch.autograd import Variable
import torch.functional as F
from torch.nn.functional import nll_loss
import dataLoader
import argparse
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import model
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io
import itertools
from datetime import datetime

parser = argparse.ArgumentParser()

# --- IO
# The locationi of training set
parser.add_argument('--imageRoot', default='/datasets/cs252csp19-public/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
parser.add_argument('--labelRoot', default='/datasets/cs252csp19-public/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
parser.add_argument('--fileList', default='/datasets/cs252csp19-public/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', help='path to input images' )
parser.add_argument('--experiment', default='train', help='the path to store sampled images and models' )
# parser.add_argument('--modelRoot', default='checkpoint', help='the path to store the testing results')
# parser.add_argument('--epochId', type=int, default=210, help='the number of epochs being trained')

# --- hyperparameters
parser.add_argument('--batchSize', type=int, default=32, help='the size of a batch' )
parser.add_argument('--nepoch', type=int, default=5, help='the training epoch')
parser.add_argument('--initLR', type=float, default=0.1, help='the initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum weight')
parser.add_argument('--iterationDecreaseLR', type=int, nargs='+', default=[1600, 2400], help='the iteration to decrease learning rate')

# --- method parameters
parser.add_argument('--imHeight', type=int, default=300, help='height of input image')
parser.add_argument('--imWidth', type=int, default=300, help='width of input image')
parser.add_argument('--numClasses', type=int, default=21, help='the number of classes' )
parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not' )
parser.add_argument('--isSpp', action='store_true', help='whether to do spatial pyramid or not' )
parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')

parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training' )
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network' )

# timer of entire script
start_time = datetime.now()
def timeElapsed():
    sec = (datetime.now() - start_time).seconds
    print('---- %.2f hour | %.2f mins' % (sec/60.0/60.0, sec/60.0))

# The detail network setting
opt = parser.parse_args()
print(opt)

colormap = io.loadmat(opt.colormap )['cmap']

# assert(opt.batchSize == 1 ), 'error occurs for mini-batch!'

if opt.isSpp == True :
    opt.isDilation = False

if opt.isDilation:
    opt.experiment += '_dilation'
    # opt.modelRoot += '_dilation'
if opt.isSpp:
    opt.experiment += '_spp'
    # opt.modelRoot += '_spp'

# Save all the codes
os.system('mkdir %s' % opt.experiment )
os.system('cp *.py %s' % opt.experiment )

if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initialize image batch
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, 300, 300) )
labelBatch = Variable(torch.FloatTensor(opt.batchSize, opt.numClasses, 300, 300) )
maskBatch = Variable(torch.FloatTensor(opt.batchSize, 1, 300, 300) )
labelIndexBatch = Variable(torch.LongTensor(opt.batchSize, 1, 300, 300) )
labelIndexBatchRaw = Variable(torch.LongTensor(opt.batchSize, 1, 300, 300) )

# Initialize network
if opt.isDilation:
    encoder = model.encoderDilation()
    decoder = model.decoderDilation()
elif opt.isSpp:
    encoder = model.encoderSPP()
    decoder = model.decoderSPP()
else:
    encoder = model.encoder()
    decoder = model.decoder()
# load pretrained weights of resnet
model.loadPretrainedWeight(encoder, isOutput=False)
# to-do: define loss layer: cross-entropy loss
# lossLayer = nn.NLLLoss
# lossLayer = nn.NLLLoss(ignore_index=255)

# encoder.load_state_dict(torch.load('%s/encoder_%d.pth' % (opt.modelRoot, opt.epochId) ) )
# decoder.load_state_dict(torch.load('%s/decoder_%d.pth' % (opt.modelRoot, opt.epochId) ) )
# encoder = encoder.eval()
# decoder = decoder.eval()

# Move network and containers to gpu
if not opt.noCuda:
    imBatch = imBatch.cuda(opt.gpuId )
    labelBatch = labelBatch.cuda(opt.gpuId )
    labelIndexBatch = labelIndexBatch.cuda(opt.gpuId )
    labelIndexBatchRaw = labelIndexBatchRaw.cuda(opt.gpuId )
    maskBatch = maskBatch.cuda(opt.gpuId )
    encoder = encoder.cuda(opt.gpuId )
    decoder = decoder.cuda(opt.gpuId )

# Initialize optimizer
# print(type(encoder.parameters()))
# print(encoder.parameters().size())
optimizer = optim.SGD(itertools.chain(encoder.parameters(), decoder.parameters()),
		      lr=opt.initLR, momentum=opt.momentum, weight_decay=5e-4 )

# Initialize dataLoader
# ------ Justin --------
# If specify width and height, then randomly crop
# ----------------------
segDataset = dataLoader.BatchLoader(
        imageRoot = opt.imageRoot,
        labelRoot = opt.labelRoot,
        fileList = opt.fileList,
        imWidth=opt.imWidth, imHeight=opt.imHeight
        )
segLoader = DataLoader(segDataset, batch_size=opt.batchSize, num_workers=0, shuffle=True )

lossArr = []
accuracyArr = []
iteration = 0
egStep = 1
set_classes = set()
for epoch in range(0, opt.nepoch ):
    ## confusion matrix of counts
    confcounts = np.zeros( (opt.numClasses, opt.numClasses), dtype=np.int64 )
    accuracy = np.zeros(opt.numClasses, dtype=np.float32 )
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    for i, dataBatch in enumerate(segLoader):
        iteration += 1

        # Read data
        image_cpu = dataBatch['im']
        imBatch.data.resize_(image_cpu.size() )
        imBatch.data.copy_(image_cpu )

        label_cpu = dataBatch['label']
        labelBatch.data.resize_(label_cpu.size() )
        labelBatch.data.copy_(label_cpu )

        labelIndex_cpu = dataBatch['labelIndex' ]
        labelIndexBatch.data.resize_(labelIndex_cpu.size() )
        labelIndexBatch.data.copy_(labelIndex_cpu )

        mask_cpu = dataBatch['mask' ]
        maskBatch.data.resize_( mask_cpu.size() )
        maskBatch.data.copy_( mask_cpu )

        labelIndexRaw_cpu = dataBatch['labelIndexRaw' ]
        labelIndexBatchRaw.data.resize_(labelIndexRaw_cpu.size() )
        labelIndexBatchRaw.data.copy_(labelIndexRaw_cpu )

        # Train network
        optimizer.zero_grad()

        ## forward
        x1, x2, x3, x4, x5 = encoder(imBatch)
        pred = decoder(imBatch, x1, x2, x3, x4, x5)
        # print(pred.size())

        ## loss & backward
        # print(type(labelIndexBatch.data))
        # loss = nll_loss(-pred, torch.squeeze(labelIndexBatch, dim=1), ignore_index=255)
        # labelIndexBatch_ = torch.tensor(labelIndexBatch, requires_grad=False)
        ## labelIndexBatch_ = labelIndexBatch.clone().detach()
        ## assert(set(maskBatch.type(torch.LongTensor).cpu().numpy().ravel()) == {0,1})
        ## labelIndexBatch_[1-maskBatch.type(torch.LongTensor)] = 255
        ## loss = lossLayer(-pred, torch.squeeze(labelIndexBatch_, dim=1))
        ### loss = lossLayer(-pred, torch.squeeze(labelIndexBatchRaw, dim=1))
        loss = torch.mean(pred * labelBatch)
        # print(pred.size(), labelIndexBatch.size())
        set_classes |= set(labelIndexBatchRaw.cpu().numpy().ravel())
        # print(type(loss))
        # print(loss.size())
        loss.backward()

        ## step
        optimizer.step()

        # Compute mean IOU (accuracy)
        # loss = torch.mean( pred * labelBatch )
        hist = utils.computeAccuracy(pred, labelIndexBatch, maskBatch )
        confcounts += hist

        for n in range(0, opt.numClasses ):
            rowSum = np.sum(confcounts[n, :] )
            colSum = np.sum(confcounts[:, n] )
            interSum = confcounts[n, n]
            accuracy[n] = float(100.0 * interSum) / max(float(rowSum + colSum - interSum ), 1e-5)

        # Output the log information
        lossArr.append(loss.cpu().data.item() )
        # np.mean(accuracy) is an accumulated accuracy within an epoch across the entire dataset
        accuracyArr.append(np.mean(accuracy))
        if iteration >= 5:
            meanLoss = np.mean(np.array(lossArr[-5:] ) )
            meanAccuracy = np.mean(np.array(accuracyArr[-5:] ) )
        else:
            meanLoss = np.mean(np.array(lossArr[:] ) )
            meanAccuracy = np.mean(np.array(accuracyArr[:] ) )
        # meanLoss = np.mean(np.array(lossArr[:] ) )
        # meanAccuracy = np.mean(accuracy )

        print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f'  \
                % ( epoch, iteration, lossArr[-1], meanLoss ) )
        print('Epoch %d iteration %d: Acc %.5f Accumulated Acc %.5f' \
                % ( epoch, iteration, accuracyArr[-1], meanAccuracy ) )
        trainingLog.write('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f \n' \
                % ( epoch, iteration, lossArr[-1], meanLoss ) )
        trainingLog.write('Epoch %d iteration %d: Acc %.5f Accumulated Acc %.5f \n' \
                % ( epoch, iteration, accuracyArr[-1], meanAccuracy ) )

        # decrease learning rate at fixed iteration
        if iteration in opt.iterationDecreaseLR:
            print('The learning rate is being decreased at iteration %d' % iteration )
            trainingLog.write('The learning rate is being decreased at iteration %d\n' % iteration )
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        # output example images and predicts every 1000 images are feed
        if iteration * opt.batchSize > egStep * 1000:
            vutils.save_image( imBatch.data , '%s/images_%d.png' % (opt.experiment, iteration ), padding=0, normalize = True)
            utils.save_label(labelBatch.data, maskBatch.data, colormap, '%s/labelGt_%d.png' % (opt.experiment, iteration ), nrows=1, ncols=1 )
            utils.save_label(-pred.data, maskBatch.data, colormap, '%s/labelPred_%d.png' % (opt.experiment, iteration ), nrows=1, ncols=1 )
            egStep += 1

    print('--------------')
    print(set_classes)
    timeElapsed()
    print('--------------')

    trainingLog.close()
    # Save the accuracy
    np.save('%s/accuracy_%d.npy' % (opt.experiment, epoch), accuracy )
    np.save('%s/accuracyArr.npy' % opt.experiment, np.array(accuracyArr) )
    np.save('%s/lossArr.npy' % opt.experiment, np.array(lossArr) )

