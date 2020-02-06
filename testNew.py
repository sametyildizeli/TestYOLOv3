import argparse
import json
import time
import numpy as np

from pathlib import Path

from models import *
from utils.datasets import *
from utils.utils import *

import pandas as pd

def test(
        cfg,
        data_cfg,
        weights,
        batch_size=1,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.2,
        nms_thres=0.3,
        save_json=False
):
    device = torch_utils.select_device()

    # Configure run
    data_cfg_dict = parse_data_cfg(data_cfg)
    print("------")
    print(data_cfg_dict)
    nC = int(data_cfg_dict['classes'])  # number of classes (80 for COCO)
    test_path = data_cfg_dict['valid']

    # Initialize model
    model = Darknet(cfg, img_size)

    print(weights.endswith('.pt'))

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)
    model.to(device).eval()
    # Get dataloader
    # dataloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path), batch_size=batch_size)
    dataloader = LoadImagesAndLabels(test_path, batch_size=batch_size, img_size=img_size)

    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    coco91class = coco80_to_coco91_class()
    stats = []

    tp_, conf_, pred_cls_, target_cls_ = [], [], [], []

    for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
        print(batch_i)
        t = time.time()
        print(paths)
        output = model(imgs.to(device))
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
        print("---TARGET:---")
        print(targets)
        print("---OUTPUT:---")
        print(output)
        print(output[0])
        print(len(output))

        print(" Prediction :")
        print(output)
        # Compute average precision for each sample
        for si, (labels, detections) in enumerate(zip(targets, output)):
            print("____ SI: ___")
            print(si)
            print("____ LABELS: ___")
            print(labels)
            print("____ DETECTIONS: ___")
            print(detections)
            
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections.cpu().numpy()
            print("**** detections: ****")
            print(detections)
            detections = detections[np.argsort(-detections[:, 4])]
            print("**** detections: ****")
            print(detections)
            print("**** detections: ****")
            
            for i in detections:
                print(i[4])

            # If no labels add number of detections as incorrect
            correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                target_cls = labels[:, 0]

                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = xywh2xyxy(labels[:, 1:5]) * img_size

                detected = []
                for *pred_bbox, conf, obj_conf, obj_pred in detections:
                    print("*pred_bbox")
                    print(*pred_bbox)
                    print("conf:")
                    print(conf)
                    print("obj_conf:")
                    print(obj_conf)
                    print("obj_pred:")
                    print(obj_pred)
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    print("pred_bbox:")
                    print(pred_bbox)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes)
                    print("iou:")
                    print(iou)
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)
            
            print("correct - pred[:,4].cpu() - pred[:,6].cpu() - tcls: ")
            detection4 = torch.from_numpy(detections[:, 4])
            detection6 = torch.from_numpy(detections[:, 6])
            print(correct)
            print(detections[:, 4])
            print(detections[:, 6])
            print(target_cls)
            print("--types--")
            print(type(correct))
            print(type(detection4))
            print(type(detection6))
            print(type(target_cls.tolist()))

            tp_.append(correct)
            conf_.append(detection4.tolist())
            pred_cls_.append(detection6.tolist())
            target_cls_.append(target_cls.tolist())

##            # Compute Average Precision (AP) per class
#    print("tp_: conf_: pred_cls_: target_cls_:")
#    print(tp_)
#    print(conf_)
#    print(pred_cls_)
#    print(target_cls_)
#-    p, r, ap, f1, ap_class = ap_per_class_new(tp_, conf_, pred_cls_, target_cls_)
    if len(tp_):
        p, r, ap, f1, ap_class = ap_per_class_new(tp_, conf_, pred_cls_, target_cls_)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        #nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    #else:
    #    nt = torch.zeros(1)

    ## Print results
    pf = '%20s' + '%10.3g' * 5  # print format
    #print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))
    print(pf % ('all', seen, mp, mr, map, mf1))

    



if __name__ == '__main__':


    allR = []
    allP = []
    
    conf_thres  =  [0.01] #np.arange(0.01, 1.01, 0.01) 
    #for i in np.linspace(0.01,1,10):
    for i in conf_thres:
        print("LOOP IS STARTED!")
        parser = argparse.ArgumentParser(prog='test.py')
        parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
        parser.add_argument('--cfg', type=str, default='cfg/urban_yolov3.cfg', help='cfg file path')
        parser.add_argument('--data-cfg', type=str, default='cfg/subt.data', help='coco.data file path')
        parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
        parser.add_argument('--conf-thres', type=float, default=i, help='object confidence threshold')
        parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')   
        parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
        parser.add_argument('--img-size', type=int, default=608, help='size of each image dimension')
        opt = parser.parse_args()
        print(opt, end='\n\n')

        with torch.no_grad():
            test(opt.cfg, opt.data_cfg, opt.weights, opt.batch_size, opt.img_size, opt.iou_thres, opt.conf_thres, opt.nms_thres, opt.save_json)
        #allR.append(mP)
        #allP.append(mR)
