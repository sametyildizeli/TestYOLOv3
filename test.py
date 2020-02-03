import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


import numpy as np
import pandas as pd
def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=608,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.3,
         save_json=False,
         model=None):
 
    print(weights + ' conf_thres: ' + str(conf_thres) + ' nms_thres: ' + str(nms_thres))
    # Initialize/load model and set device
    
    if model is None:
        device = torch_utils.select_device(opt.device)
        verbose = True

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False
    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    test_path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min([os.cpu_count(), batch_size, 16]),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists( weights + 'test_batch0''.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs

        # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(d[6])],
                                  'bbox': [floatn(x, 3) for x in box[di]],
                                  'score': floatn(d[4], 5)})

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    
    ################## MY PART ############

    df = pd.DataFrame(columns=['Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1','weights','threshold(iou-conf-nms)'])
    weightsName  = [weights]
    class_       = ['all']
    image_       = [seen]
    targets_     = [nt.sum()]
    p_           = [mp]
    r_           = [mr]
    mAP_         = [map]
    f1_          = [mf1]
    thresholding = [str(iou_thres) + " " + str(conf_thres) + " " + str(nms_thres)]

    #######################################

    print(class_)
    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
            class_.append(names[c])
            image_.append(seen)
            targets_.append(nt[c])
            p_.append(p[i])
            r_.append(r[i])
            mAP_.append(ap[i])
            f1_.append(f1[i])
            weightsName.append(weights)
            thresholding.append(str(iou_thres) + " " + str(conf_thres) + " " + str(nms_thres))
    df["Class"] = class_
    df["Images"] = image_
    df["Targets"] = targets_
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
    
    tableName_    =  'TestResults/lastTablesv3/'+ weightsSplit2 
    
    if not os.path.exists(tableName_):
        os.makedirs(tableName_)

    df.to_csv(tableName_ + str(int(100*conf_thres)) + '-table-' + weightsSplit + 'conf-' + str(conf_thres) + 'nms-' + str(nms_thres) + '.csv')
  
    # Save JSON
    if save_json and map and len(jdict):
        try:
            imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
            with open('results.json', 'w') as file:
                json.dump(jdict, file)

            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map = cocoEval.stats[1]  # update mAP to pycocotools mAP
        except:
            print('WARNING: missing dependency pycocotools from requirements.txt. Can not compute official COCO mAP.')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/4my_yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    #parser.add_argument('--weights', type=str, default='weights4_16/best.pt', help='path to weights file')
    parser.add_argument('--batch-size', type=int, default=12, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    #parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    #parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)


    weights     =  ['weights2_16/backup10.pt','weights2_16/backup50.pt','weights2_16/backup100.pt','weights2_16/backup150.pt','weights2_16/best.pt']
  
    conf_thres  =  np.arange(0, 1.01, 0.01) 
 
    ##print(conf_thres)

    nms_thres   =  [0.3] #[0.2, 0.3, 0.4, 0.5]

    for i in range(0,len(weights)):
        for k in range(0,len(conf_thres)):
            for l in range(0,len(nms_thres)):
                with torch.no_grad():
                    test(opt.cfg,
                         opt.data,
                         weights[i],
                         opt.batch_size,
                         opt.img_size,
                         opt.iou_thres,
                         conf_thres[k],
                         nms_thres[l],
                         opt.save_json)
