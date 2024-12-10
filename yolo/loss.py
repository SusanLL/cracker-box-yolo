"""
CS 6375 Homework 2 Programming
Implement the compute_loss() function in this python script
"""
import os
import torch
import torch.nn as nn


def compute_iou(pred, gt):
    x1p = pred[0] - pred[2] * 0.5
    x2p = pred[0] + pred[2] * 0.5
    y1p = pred[1] - pred[3] * 0.5
    y2p = pred[1] + pred[3] * 0.5
    areap = (x2p - x1p + 1) * (y2p - y1p + 1)    
    
    x1g = gt[0] - gt[2] * 0.5
    x2g = gt[0] + gt[2] * 0.5
    y1g = gt[1] - gt[3] * 0.5
    y2g = gt[1] + gt[3] * 0.5
    areag = (x2g - x1g + 1) * (y2g - y1g + 1)

    xx1 = max(x1p, x1g)
    yy1 = max(y1p, y1g)
    xx2 = min(x2p, x2g)
    yy2 = min(y2p, y2g)

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / (areap + areag - inter)    
    return iou


def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
    batch_size = output.shape[0]
    num_grids = output.shape[2]
    # Compute mask with shape (batch_size, num_boxes, 7, 7) for box assignment
    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids, device=output.device)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids, device=output.device)

    # Compute assignment of predicted bounding boxes for ground truth bounding boxes
    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
                # If the gt mask is 1
                if gt_mask[i, j, k] > 0:
                    # Transform gt box
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * grid_size
                    gt[1] = gt[1] * grid_size + j * grid_size
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size

                    select = 0
                    max_iou = -1
                    # Select the one with maximum IoU
                    for b in range(num_boxes):
                        # Center x, y and width, height
                        pred = pred_box[i, 5 * b:5 * b + 4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b
                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou
                    print('select box %d with iou %.2f' % (select, max_iou))

    # Weights for different components of the loss
    weight_coord = 5.0
    weight_noobj = 0.5

    # Compute each component of the YOLO loss
    # Loss for x and y coordinates
    loss_x = torch.sum(
        weight_coord * box_mask * torch.pow(output[:, 0:5*num_boxes:5] - gt_box[:, 0:1, :, :], 2.0)
    )
    loss_y = torch.sum(
        weight_coord * box_mask * torch.pow(output[:, 1:5*num_boxes:5] - gt_box[:, 1:2, :, :], 2.0)
    )

    # Loss for width and height
    loss_w = torch.sum(
        weight_coord * box_mask * torch.pow(
            torch.sqrt(torch.clamp(output[:, 2:5*num_boxes:5], min=1e-6)) -
            torch.sqrt(torch.clamp(gt_box[:, 2:3, :, :], min=1e-6)),
            2.0
        )
    )
    loss_h = torch.sum(
        weight_coord * box_mask * torch.pow(
            torch.sqrt(torch.clamp(output[:, 3:5*num_boxes:5], min=1e-6)) -
            torch.sqrt(torch.clamp(gt_box[:, 3:4, :, :], min=1e-6)),
            2.0
        )
    )

    # Loss for objects (confidence)
    loss_obj = torch.sum(
        box_mask * torch.pow(box_confidence - output[:, 4:5*num_boxes:5], 2.0)
    )

    # Loss for no-object cells
    loss_noobj = torch.sum(
        weight_noobj * (1 - box_mask) * torch.pow(output[:, 4:5*num_boxes:5], 2.0)
    )

    # Classification loss
    loss_cls = torch.sum(
        gt_mask.unsqueeze(1) * torch.pow(output[:, 5*num_boxes:] - gt_box[:, 4:, :, :], 2.0)
    )

    # Compute total loss
    loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls

    # Debugging output
    print(
        'lx: %.4f, ly: %.4f, lw: %.4f, lh: %.4f, lobj: %.4f, lnoobj: %.4f, lcls: %.4f' %
        (loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_obj.item(), loss_noobj.item(), loss_cls.item())
    )

    return loss
