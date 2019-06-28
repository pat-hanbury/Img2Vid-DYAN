import os
import random

import yaml
from natsort import natsorted

import torch
import torch.nn as nn

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from PerceptualSimilarity.models import dist_model as dm
from DYAN.Code.DyanOF import OFModel
from seg2vid.src.utils.ops import flowwrapper


# modified from original DYAN function
def loadOpticalFlowModel(dyan_lambda, ckpt_file, gpu_id=0):
    checkpoint = torch.load(ckpt_file)
    stateDict = checkpoint['state_dict']
    
    FRA = 3 # if Kitti: FRA = 9
    PRE = 1 # represents predicting 1 frame
    
    # load parameters
    Dtheta = stateDict['l1.theta'] 
    Drr    = stateDict['l1.rr']
    model = OFModel(dyan_lambda, Drr, Dtheta, FRA,PRE,gpu_id)
    model.cuda(gpu_id)

    return model

def get_model(dyan_lambda):
    with open('DYAN/Code/configs.yaml', 'r') as file:
        configs = yaml.safe_load(file)
        
    ckpt_file = configs['opticalflow_ckpt_file']
    
    return loadOpticalFlowModel(dyan_lambda, ckpt_file)

def get_data(root_dir):
    files = os.listdir(root_dir)
    gt_img_pths = natsorted([os.path.join(root_dir,x) for x in files if x.startswith("data")])
    flow_pths = natsorted([os.path.join(root_dir,x) for x in files if x.startswith("forward")])
    
    gt_imgs = [np.load(pth)[0] for pth in gt_img_pths]
    flows = [np.load(pth)[0] for pth in flow_pths]
    
    assert len(gt_imgs) == len(flows), "Error: number of flows and GT data differ"
    
    return gt_imgs, flows

def run_test(model, scale_factor, is_consecutive, gen_inter_frames, gt_imgs, flows):
    predictions = [] # a list of lists of images (could be [[img], [img]] or [[img, img], [img, img]]) 
    total_MSE_loss = 0
    total_LPIPS_loss = 0
    
    MSE_loss = nn.MSELoss()
    LPIPS_loss = dm.DistModel()
    LPIPS_loss.initialize(model='net-lin',net='vgg',use_gpu=True,spatial=True)
    
    model.cuda()
    
    for gt, flow in zip(gt_imgs, flows):
        flow = convert_flow(flow, scale_factor, is_consecutive)
        flow = flow.cuda()
        dyan_out = model.forward(flow)
        flow_predicitons = dyan_out.data.resize(4,2,256,256)
        
        frame0 = torch.from_numpy(gt[0]) # first frame
        
        if is_consecutive and gen_inter_frames:
            # loop through four optical flows
            output_frames = []
            warped_frame = frame0.unsqueeze(0).cuda()
            for i in range(4):
                current_flow = flow_predicitons[i].unsqueeze(0).cuda()
                warped_frame = warp(warped_frame, current_flow)
                gt_frame = torch.from_numpy(gt[i+1]).unsqueeze(0).cuda()
                
                if i == 3:
                    total_MSE_loss+= MSE_loss(warped_frame, gt_frame)
                    total_LPIPS_loss+= LPIPS_loss.forward(warped_frame, gt_frame)
                output_frames.append(warped_frame)
        else:
            frame_indx = 3 if is_consecutive else 0
            input_frame = torch.from_numpy(np.expand_dims(gt[frame_indx], axis=0)).cuda()
            current_flow = flow_predicitons[3].unsqueeze(0).cuda()
            warped_frame = warp(input_frame, current_flow)
            
            gt_frame = torch.from_numpy(gt[4]).unsqueeze(0).cuda()
            
            total_MSE_loss+= MSE_loss(warped_frame, gt_frame)
            total_LPIPS_loss+= LPIPS_loss.forward(warped_frame, gt_frame)
            output_frames = [warped_frame]
            
        predictions.append(output_frames)
    
    avg_mse = total_MSE_loss / len(gt_imgs)
    avg_lpips = (total_LPIPS_loss / len(gt_imgs)).mean()
    
    return predictions, avg_mse, avg_lpips
        
            
            
            
        
def convert_flow(flow, scale_factor, is_consecutive):
    """
    Converts flow from Seg2Vid format to
    DYAN format
    """
    w, h = (flow.shape[2], flow.shape[3])
    flow = flow.transpose(1,0,2,3).reshape(2, 4, w*h)
    flow = torch.from_numpy(flow[:,:-1])
    
    if is_consecutive:
        for i in range(len(flow) - 1):
            idx = len(flow) - i - 1
            flow[idx] = flow[idx] - flow[idx-1]
               
    return flow*scale_factor

def warp(input,tensorFlow):
    torchHorizontal = torch.linspace(-1.0, 1.0, input.size(3))
    torchHorizontal = torchHorizontal.view(1, 1, 1, input.size(3)).expand(input.size(0), 1, input.size(2), input.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, input.size(2))
    torchVertical = torchVertical.view(1, 1, input.size(2), 1).expand(input.size(0), 1, input.size(2), input.size(3))

    tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda()
    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=input, grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
    
    
def print_results(MSE_loss, LPIPS_loss, lmbda, 
                  scale, is_consecutive, gen_inter):
    OF_format = "Consecutive" if is_consecutive else "Cumulative"
    
    print("****"*15)
    print("Configurations: ")
    print(f"DYAN Lambda: {lmbda}")
    print(f"Scale Factor: {scale}")
    print(f"Optical Flow Format: {OF_format}")
    print(f"Intermediate Frame Generation: {gen_inter}\n")
    print(f"Average MSE loss:   {MSE_loss:.4f}")
    print(f"Average LPIPS loss: {LPIPS_loss:.4f}")
    
def display_random_examples(preds, gts, num_samples=1):
    for i in range(num_samples):
        indx = random.randint(0, len(preds)-1)
        pred = preds[indx]
        gt = gts[indx]
        
        plt.xticks([])
        plt.yticks([])

        
        # print GT sequence
        # mpl.rcParams['figure.dpi'] = 300
        img0 = gt[0].transpose(1,2,0)
        for img in gt[1:]:
            img0 = np.hstack((img0, img.transpose(1,2,0)))
        plt.imshow(img0)
        plt.title("Ground Truth Sequence")
        plt.show()
        
        if len(pred) == 1:
            plt.xticks([])
            plt.yticks([])
            
            # mpl.rcParams['figure.dpi'] = 100
            pred = pred[0].cpu().numpy()[0]
            gt = gt[-1]
            img = np.hstack((gt.transpose(1,2,0), pred.transpose(1,2,0)))
            plt.imshow(img)
            plt.title("Left: GT 5th Frame -- Right: Predicted Frame 5")
            plt.show()

        else:
            plt.xticks([])
            plt.yticks([])
            
            img0 = pred[0][0].cpu().numpy().transpose(1,2,0)
            for img in pred[1:]:
                img = img[0].cpu().numpy().transpose(1,2,0)
                img0 = np.hstack((img0, img))
            plt.imshow(img0)
            plt.title("Predicted Sequence")
            plt.show()

    print("****"*15)
    
def test_seg2vid(gt_imgs, flows):
    flow_wrapper = flowwrapper()
    predictions = []
    
    total_MSE_loss = 0
    total_LPIPS_loss = 0
    
    MSE_loss = nn.MSELoss()
    LPIPS_loss = dm.DistModel()
    LPIPS_loss.initialize(model='net-lin',net='vgg',use_gpu=True,spatial=True)
    

    for gt, flow in zip(gt_imgs, flows):
        frame0 = torch.from_numpy(gt[0]).unsqueeze(0)
        frame4 = torch.from_numpy(gt[4]).unsqueeze(0)
        
        flow = torch.from_numpy(flow).unsqueeze(0)
        
        seg2vid_preds = [torch.unsqueeze(flow_wrapper(frame0, flow[:, :, i, :, :]), 1) for i in range(4)]
        
        pred = seg2vid_preds[0][0]
        
        total_MSE_loss+= MSE_loss(pred, frame4)
        total_LPIPS_loss+= LPIPS_loss.forward(pred, frame4)
        
        predictions.append(seg2vid_preds)
    
    return predictions, (total_MSE_loss/len(gt_imgs)).data, (total_LPIPS_loss/len(gt_imgs)).mean()

def print_seg2vid_results(MSE_loss, LPIPS_loss):
    print("****"*15)
    print(f"Average MSE loss:   {MSE_loss:.4f}")
    print(f"Average LPIPS loss: {LPIPS_loss:.4f}")
    
def display_seg2vid_example(preds, gts, num_samples=1, sample_num=None):
    for i in range(num_samples):
        if sample_num is not None:
            indx = sample_num
        else:
            indx = random.randint(0, len(preds)-1)
        pred = preds[indx]
        gt = gts[indx]
    
        plt.xticks([])
        plt.yticks([])

        
        # print GT sequence
        # mpl.rcParams['figure.dpi'] = 300
        img0 = gt[0].transpose(1,2,0)
        for img in gt[1:]:
            img0 = np.hstack((img0, img.transpose(1,2,0)))
        plt.imshow(img0)
        plt.title("Ground Truth Sequence")
        plt.show()
        
        img0 = pred[0][0][0]
        img0 = img.transpose(1,2,0)
        for img in pred[1:]:
            img = img[0][0]
            img = img.cpu().numpy().transpose(1,2,0)
            img0 = np.hstack((img0, img))
            
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img0)
        plt.title("Seg2Vid Prediction")
        plt.show()
        