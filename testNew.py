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
    print(weights + ' conf_thres: ' + str(conf_thres) + ' nms_thres: ' + str(nms_thres))

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
        t = time.time()

        output = model(imgs.to(device))
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)

        # Compute average precision for each sample
        for si, (labels, detections) in enumerate(zip(targets, output)):            
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections.cpu().numpy()
            detections = detections[np.argsort(-detections[:, 4])]
            
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

                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)

                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes)

                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)
            
            detection4 = torch.from_numpy(detections[:, 4])
            detection6 = torch.from_numpy(detections[:, 6])

            tp_.append(correct)
            conf_.append(detection4.tolist())
            pred_cls_.append(detection6.tolist())
            target_cls_.append(target_cls.tolist())

##            # Compute Average Precision (AP) per class

    if len(tp_):
        p, r, ap, f1, ap_class = ap_per_class_new(tp_, conf_, pred_cls_, target_cls_)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

    ## Print results
    pf = '%20s' + '%10.3g' * 5  # print format
    print(pf % ('all', seen, mp, mr, map, mf1))


    df = pd.DataFrame(columns=['Class', 'Images', 'P', 'R', 'mAP@0.5', 'F1','weights','threshold(iou-conf-nms)'])
    weightsName  = [weights]
    class_       = ['all']
    image_       = [seen]
    p_           = [mp]
    r_           = [mr]
    mAP_         = [map]
    f1_          = [mf1]
    thresholding = [str(iou_thres) + " " + str(conf_thres) + " " + str(nms_thres)]


    df["Class"] = class_
    df["Images"] = image_
    df["P"] = p_
    df["R"] = r_
    df["mAP@0.5"] = mAP_
    df["F1"] = f1_
    df["weights"] = weightsName 
    df['threshold(iou-conf-nms)'] = thresholding



    
    weightsSplit = weights.split('/')
    weightsSplit = weightsSplit[0]
    
    weightsSplit2 = weights.split('.')
    weightsSplit2 = weightsSplit2[0] + '/'
    
    tableName_    =  'TestResults/lastTablesv2/'+ weightsSplit2 
    
    if not os.path.exists(tableName_):
        os.makedirs(tableName_)

    df.to_csv(tableName_ + str(int(100*conf_thres)) + '-table-' + weightsSplit + 'conf-' + str(conf_thres) + 'nms-' + str(nms_thres) + '.csv')
  

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(prog='testv4.py')
    parser.add_argument('--batch-size', type=int, default=12, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/urban_yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/subt.data', help='coco.data file path')
    #parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    #parser.add_argument('--conf-thres', type=float, default=i, help='object confidence threshold')
    #parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')   
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=608, help='size of each image dimension')
    opt = parser.parse_args()
    print(opt, end='\n\n')


    weights     =  ["weights/best.pt"]

    conf_thres  =  np.arange(0.01, 1.01, 0.01) 
 
    ##print(conf_thres)

    nms_thres   =  [0.3]

    for i in range(0,len(weights)):
        for k in range(0,len(conf_thres)):
            for l in range(0,len(nms_thres)):
                with torch.no_grad():
                    test(opt.cfg,
                         opt.data_cfg,
                         weights[i],
                         opt.batch_size,
                         opt.img_size,
                         opt.iou_thres,
                         conf_thres[k],
                         nms_thres[l],
                         opt.save_json)
