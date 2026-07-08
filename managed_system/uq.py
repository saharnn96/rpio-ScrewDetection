#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/10 下午2:35
# @Author  : Chengjie Lu
# @File    : inference_uq.py
# @Software: PyCharm
import json

import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch import nn
from torchvision.ops import DropBlock2d
from ultralytics import YOLO
import cv2
from pathlib import Path
import os
from deepluq.utils import DBSCANCluster, wbf_clustering
from deepluq import metrics_dl
import time
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from tqdm import tqdm
from itertools import product

import warnings

warnings.filterwarnings("ignore")


# Load a YOLO model from the given path
def load_model(model_path):
    model = YOLO(model_path)
    return model


# Draw bounding boxes and labels on the image for each detected object
def draw_boxes_on_image(image, boxes, target_boxes, model_names):
    for box in boxes:
        # print(box.xyxy[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
        # print(x1, y1, x2, y2)
        class_id = int(box.cls)  # Get class id
        class_name = model_names[class_id]  # Get class name
        confidence = box.conf[0]  # Get confidence score
        label = f"{class_name}: {confidence:.2f}"  # Prepare label text
        # Set color based on class id
        color = (0, 0, 255) if class_id == 0 else (255, 255, 0) if class_id == 1 else (0, 255, 0)
        # Draw label at different positions based on class
        if class_id == 0:
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif class_id == 1:
            cv2.putText(image, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif class_id == 2:
            cv2.putText(image, label, (x1 + 20, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Draw rectangle for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    i = 0
    for box in target_boxes:
        #print(box)
        color = (0, 255, 255) if i == 0 else (0, 0, 0)

        x1, y1, x2, y2 = map(int, box)
        #print(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        i += 1
    return image


# Run inference on a set of test images and optionally save the results
def predict_test_set(model_path, test_images_path, save=False, output_path=None):
    # Load the YOLO model
    model = load_model(model_path)
    model.info()

    # Keep BatchNorm layers in eval mode for stability
    for m in model.model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train()

    # Collect all .png and .jpg images in the test directory
    test_images = list(test_images_path.glob('*.png')) + list(test_images_path.glob('*.jpg'))

    # #print(test_images)
    # Run model prediction on the test images
    results = model.predict(source=[test_images[0]], show=False, show_labels=False, save=False, verbose=False)

    target_path = test_images[0].parent.parent / 'labels' / test_images[0].name
    target_path.with_suffix(".txt")
    target = yolo_to_absolute(read_yolo_file(target_path.with_suffix(".txt")))

    # #print(results[0].boxes)
    # Iterate through each result and process the images
    for idx, result in enumerate(results):
        # Draw bounding boxes and labels on the original image
        img_with_boxes = draw_boxes_on_image(result.orig_img, result.boxes, target['boxes'], model.names)
        # Display the image with bounding boxes
        cv2.imshow('Image with Bounding Boxes', img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # If saving is enabled and output path is provided, save the image
        if save and output_path:
            # Create output directory if it doesn't exist
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            out_file = os.path.join(output_path, f"pred_{idx}.jpg")
            cv2.imwrite(str(out_file), img_with_boxes)

    # Return the results for further processing if needed
    return results


# Run real-time inference using the webcam
def webcam_inference(model_path):
    # Load the YOLO model
    model = load_model(model_path)
    # Open the default webcam (device 0)
    cap = cv2.VideoCapture(0)
    #print("Press 'q' to quit. Press 's' to save the current frame with prediction.")
    save_count = 0
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            break  # Exit loop if frame not captured
        # Run prediction on the current webcam frame
        results = model.predict(source=[frame], show=False, show_labels=False, save=False, verbose=False)
        result = results[0]
        # Draw bounding boxes and labels on the frame
        img_with_boxes = draw_boxes_on_image(frame.copy(), result.boxes, model.names)
        # Display the frame with predictions
        cv2.imshow('Webcam Inference', img_with_boxes)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Quit if 'q' is pressed
        elif key == ord('s'):
            # Save the current frame with predictions if 's' is pressed
            filename = f"webcam_pred_{save_count}.jpg"
            cv2.imwrite(filename, img_with_boxes)
            #print(f"Saved: {filename}")
            save_count += 1
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows


def postprocess(output):
    vr = []
    entropy = []
    mi = []
    tv_box = []
    ps = []

    for key, val in output.items():
        # Collect metrics
        vr.append(val['detection']['variation_ratio [classification]'])
        entropy.append(val['detection']['entropy [classification]'])
        mi.append(val['detection']['mutual_info [classification]'])
        tv_box.append(val['detection']['total_var_box [regression]'])
        ps.append(val['detection']['predictive_surface [regression]'])

    # Aggregate metrics (vectorized)
    output['Metrics_Avg'] = {
        'variation_ratio [classification]': float(np.mean(vr)),
        'entropy [classification]': float(np.mean(entropy)),
        'mutual_info [classification]': float(np.mean(mi)),
        'total_var_box [regression]': float(np.mean(tv_box)),
        'predictive_surface [regression]': float(np.mean(ps)),
    }

    return output


def yolo_to_absolute(yolo_labels, img_w=1280, img_h=736):
    """
    Convert YOLO normalized bounding boxes to absolute coordinates and compute areas.
    Vectorized for speed.
    """

    if len(yolo_labels) == 0:
        # handle empty input
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.long),
            'areas': torch.zeros((0,), dtype=torch.float32)
        }

    yolo_tensor = torch.tensor(yolo_labels, dtype=torch.float32)  # shape [N, 5]

    labels = yolo_tensor[:, 0].long()  # class IDs
    x_c = yolo_tensor[:, 1] * img_w  # center x
    y_c = yolo_tensor[:, 2] * img_h  # center y
    w = yolo_tensor[:, 3] * img_w  # width
    h = yolo_tensor[:, 4] * img_h  # height

    # convert to [xmin, ymin, xmax, ymax]
    half_w = w / 2
    half_h = h / 2

    boxes = torch.stack([
        x_c - half_w,
        y_c - half_h,
        x_c + half_w,
        y_c + half_h
    ], dim=1)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return {
        'boxes': boxes,
        'labels': labels,
        'areas': areas
    }


# Function to read a YOLO label file into a list of lists
def read_yolo_file(file_path):
    """
    Reads a YOLO .txt label file and converts it into a list of lists.

    Each line in the file should have the format:
        class_id x_center y_center width height
    with coordinates normalized between 0 and 1.

    Returns:
        list of lists: [[class_id, x_center, y_center, width, height], ...]
    """
    yolo_labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip invalid lines
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            yolo_labels.append([class_id, x_center, y_center, width, height])
    return yolo_labels


def run_uq(model, test_image, T, uq_method, uq_config):#, map_overall, metrics_dict, uq_metrics, dict_fn):
    # Load the YOLO model
    # model = load_model(model_p)
    # model.info()
    if uq_method == 'mc_dropout':
        model.model.drop = nn.Dropout(uq_config[0])
    elif uq_method == 'mc_dropblock':
        model.model.drop = DropBlock2d(block_size=uq_config[1], p=uq_config[0])
    uq = metrics_dl.DLMetrics()
    predictions, pred_id, mc_locations = {}, 0, []

    # target_path = test_image[0].parent.parent / 'labels' / test_image[0].name
    # target_path.with_suffix(".txt")
    # target = yolo_to_absolute(read_yolo_file(target_path.with_suffix(".txt")))
    #
    # map_metric = torchmetrics.detection.MeanAveragePrecision(max_detection_thresholds=[1, 5, 100],
    #                                                          iou_thresholds=[0.5])

    for i in range(T):
        # Run model prediction on the test images
        preds = model.predict(source=test_image, show=False, show_labels=False, save=False, verbose=False)

        boxes, logits_l, scores, labels = [], [], [], []

        prediction_t, pred_id_t = {}, 0

        # #print('detection: ', preds[0].detection)
        # #print('boxes: ', preds[0].boxes)
        for detection, box_n in zip(preds[0].detection.cpu().detach().numpy(),
                                    preds[0].boxes.xyxyn.cpu().detach().numpy()):
            # #print(detection, box_n)
            box = detection[:4]
            logits = detection[4:]
            score = max(logits)
            label = torch.argmax(torch.from_numpy(logits))
            # preds = [{'boxes': preds[0], 'scores': preds[1], 'labels': preds[2]}]
            boxes.append(box)
            logits_l.append(logits)
            scores.append(score)
            labels.append(label)

            mc_locations.append(
                np.concatenate((box, np.array([int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)])),
                               axis=None))

            predictions.update({'label_{}'.format(pred_id):
                {
                    'box': box.tolist(),
                    'box_n': box_n.tolist(),
                    'label': label.tolist(),
                    'score': score.tolist(),
                    'logit': logits.tolist(),
                }
            })
            pred_id += 1

            """
            start:
            save for each prediction. comment it if need faster speed
            """
            prediction_t.update({'label_{}'.format(pred_id_t):
                {
                    'box': box.tolist(),
                    'label': label.tolist(),
                    'score': score.tolist(),
                    'logit': logits.tolist(),
                }
            })
            pred_id_t += 1

        # pred_t_dict = dict_fn / f'prediction_{i}.json'

        # with open(pred_t_dict, 'w') as f:
        #     json.dump(prediction_t, f, indent=4)
        # end

        """
        start:
        preparation for map calculation
        """
        preds_dict = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'scores': torch.tensor(scores, dtype=torch.float32),
            'logits': torch.tensor(logits_l, dtype=torch.float32),
        }

        # #print(preds_dict, target)
        # map_metric.update([preds_dict], [target])
        # map_overall.update([preds_dict], [target])
        # end

    # #print(predictions)

    predictions = wbf_clustering(
        predictions,
        iou_thr=0.5,
        skip_box_thr=0.01
    )

    #print(predictions)

    # dbscan_cluster = DBSCANCluster(x=mc_locations)
    # predictions = dbscan_cluster.cluster_preds(preds=predictions)
    #
    # #print(predictions)

    for key in predictions.keys():
        # self.put_text(key, predictions[key]['box'][0], image_og)
        logit_sample_trans = np.transpose(predictions[key]['logit'])
        vr = uq.cal_vr(predictions[key]['logit'])
        shannon_entropy = uq.calcu_entropy(np.mean(logit_sample_trans, axis=1))
        mi = uq.calcu_mi(predictions[key]['logit'])
        tv_box = uq.calcu_tv(predictions[key]['box'], tag='bounding_box')
        predictive_surface = uq.calcu_prediction_surface(predictions[key]['box'])

        ''''
        get avg prediction and uncertainty metrics
        '''

        predictions[key]['detection'].update(
            {
                'detection times (out of {})'.format(T): len(predictions[key]['score']),
                'variation_ratio [classification]': vr,
                'entropy [classification]': shannon_entropy,
                'mutual_info [classification]': mi,
                'total_var_box [regression]': tv_box,
                'predictive_surface [regression]': predictive_surface
            }
        )

    """
    start:
    calculate avg uq metrics, uncomment it if needed.
    """
    predictions = postprocess(predictions)
    # end

    # #print(predictions)

    # clustered_dict = dict_fn / 'clustered_predictions.json'

    # with open(clustered_dict, 'w') as f:
    #     json.dump(predictions, f, indent=4)

    # """
    # start:
    # preparation for map calculation
    # """
    # map_results = map_metric.compute()
    # m = [test_image[0].stem, test_image[0].stem, map_results["map"].item(),
    #      predictions['Metrics_Avg']['variation_ratio [classification]'],
    #      predictions['Metrics_Avg']['entropy [classification]'],
    #      predictions['Metrics_Avg']['mutual_info [classification]'],
    #      predictions['Metrics_Avg']['total_var_box [regression]'],
    #      predictions['Metrics_Avg']['predictive_surface [regression]'],
    #      map_results["map_50"].item(),
    #      map_results["map_75"].item(),
    #      map_results["map_small"].item(),
    #      map_results["map_medium"].item(),
    #      map_results["map_large"].item(),
    #      map_results["mar_1"].item(),
    #      map_results["mar_5"].item(),
    #      map_results["mar_100"].item(),
    #      map_results["mar_small"].item(),
    #      map_results["mar_medium"].item(),
    #      map_results["mar_large"].item()
    #      ]
    # for t, key in enumerate(list(metrics_dict.keys())):
    #     metrics_dict[list(metrics_dict.keys())[t]].append(m[t])
    #
    # uq_metrics['vr'].append(predictions['Metrics_Avg']['variation_ratio [classification]'])
    # uq_metrics['ie'].append(predictions['Metrics_Avg']['entropy [classification]'])
    # uq_metrics['mi'].append(predictions['Metrics_Avg']['mutual_info [classification]'])
    # uq_metrics['tr'].append(predictions['Metrics_Avg']['total_var_box [regression]'])
    # uq_metrics['ps'].append(predictions['Metrics_Avg']['predictive_surface [regression]'])
    # #  end
    return predictions


def uq_main(uq_method, uq_config, test_data, model_version, n_predictions):
    # Define the model path
    model_path = 'uq_evaluation/models/{}.pt'.format(model_version)
    model = load_model(model_path)
    # Define the path to the test images
    test_path = Path('uq_evaluation/origimg/{}'.format(test_data))
    test_images_path = test_path / 'images'
    test_images = list(test_images_path.glob('*.png')) + list(test_images_path.glob('*.jpg'))

    map_metric_overall = torchmetrics.detection.MeanAveragePrecision(max_detection_thresholds=[1, 5, 100],
                                                                     iou_thresholds=[0.5])

    metrics_items = {'image': [], 'image_name': [], 'MAP': [], 'UQ[VR]': [], 'UQ[IE]': [], 'UQ[MI]': [], 'UQ[TR]': [],
                     'UQ[PS]': [],
                     'MAP[50]': [], 'MAP[75]': [], 'MAP[Small]': [], 'MAP[Medium]': [], 'MAP[Large]': [], 'MAR[1]': [],
                     'MAR[5]': [],
                     'MAR[100]': [], 'MAR[Small]': [], 'MAR[Medium]': [], 'MAR[Large]': []}

    uq_metrics = {'vr': [], 'ie': [], 'mi': [], 'tr': [], 'ps': []}

    headers = ['image', 'image_name', 'MAP', 'UQ[VR]', 'UQ[IE]', 'UQ[MI]', 'UQ[TR]', 'UQ[PS]',
               'MAP[50]', 'MAP[75]', 'MAP[Small]', 'MAP[Medium]', 'MAP[Large]', 'MAR[1]', 'MAR[5]',
               'MAR[100]', 'MAR[Small]', 'MAR[Medium]', 'MAR[Large]']

    folder = Path('experiment_results_/experiment_results_{}/{}'.format(uq_method, model_version))
    dataset_folder = folder / 'dataset/{}'.format(test_data)
    dataset_folder.mkdir(parents=True, exist_ok=True)

    log_fn = folder / 'logs_{}_origimg-{}.csv'.format(uq_config[0], test_data) if uq_method == 'mc_dropout' else \
        folder / 'logs_{}_{}_origimg-{}.csv'.format(uq_config[0], uq_config[1], test_data)

    pd.DataFrame([headers]).to_csv(log_fn, mode='w', header=False, index=False)

    # Predict on the test set
    for t_image in test_images:
        dict_path = dataset_folder / '{}_{}_{}'.format(uq_method, uq_config[0], t_image.stem) \
            if uq_method == 'mc_dropout' else \
            dataset_folder / '{}_{}_{}_{}'.format(uq_method, uq_config[0], uq_config[1], t_image.stem)
        dict_path.mkdir(parents=True, exist_ok=True)

        try:
            run_uq(model, [t_image], T=n_predictions, uq_method=uq_method, uq_config=uq_config,
                   map_overall=map_metric_overall, metrics_dict=metrics_items, uq_metrics=uq_metrics, dict_fn=dict_path)
        except:
            #print(f'\nNothing detected for: {t_image}')
            pd.DataFrame([[-1, t_image.stem]]).to_csv(log_fn, mode='a', header=False, index=False)

    """
    start:
    preparation for map calculation
    """
    # map_results_all = map_metric_overall.compute()
    # m = [len(test_images), 'overall', map_results_all['map'].item(),
    #      sum(uq_metrics['vr']) / len(uq_metrics['vr']),
    #      sum(uq_metrics['ie']) / len(uq_metrics['ie']),
    #      sum(uq_metrics['mi']) / len(uq_metrics['mi']),
    #      sum(uq_metrics['tr']) / len(uq_metrics['tr']),
    #      sum(uq_metrics['ps']) / len(uq_metrics['ps']),
    #      map_results_all["map_50"].item(),
    #      map_results_all["map_75"].item(),
    #      map_results_all["map_small"].item(),
    #      map_results_all["map_medium"].item(),
    #      map_results_all["map_large"].item(),
    #      map_results_all["mar_1"].item(),
    #      map_results_all["mar_5"].item(),
    #      map_results_all["mar_100"].item(),
    #      map_results_all["mar_small"].item(),
    #      map_results_all["mar_medium"].item(),
    #      map_results_all["mar_large"].item()
    #      ]

    # for t, key in enumerate(list(metrics_items.keys())):
    #     metrics_items[list(metrics_items.keys())[t]].append(m[t])
    #
    # df = pd.DataFrame(metrics_items)
    # df.to_csv(log_fn, mode='a', header=False, index=False)
    # # end


if __name__ == '__main__':
    # uq_ms = ['mc_dropout', 'mc_dropblock']
    d_rates = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    b_sizes = [1, 3, 5, 7, 9]
    # test_datasets = ['orig', 'adv_run_1', 'adv_run_2', 'adv_run_3', 'adv_run_4', 'adv_run_5', 'adv_run_6',
    # 'adv_run_7', 'adv_run_8', 'adv_run_9', 'adv_run_10']
    test_datasets = ['new']
    # models = ['best_initial', 'best_added_more_noscrews_at_diff_exposure', 'best_add_screw_fixture']
    models = ['best_added_more_noscrews_at_diff_exposure']

    uq_m = 'mc_dropout'
    for model_n, test_dataset, d_rate in tqdm(product(models, test_datasets, d_rates),
                                              total=len(models) * len(test_datasets) * len(d_rates),
                                              desc="Running UQ experiments"):
        #print(f"\n 🔍 UQ method: {uq_m} | 🧠 Model: {model_n} | 📊 Dataset: {test_dataset} "
        #      f"| 💧 Dropout rate: {d_rate} \n")

        uq_main(uq_method=uq_m, uq_config=[d_rate], test_data=test_dataset, model_version=model_n, n_predictions=10)

    # """
    # mc dropblock
    # """
    # uq_ms = ['mc_dropblock']
    # for uq_m in uq_ms:
    #     for model_n, test_dataset, d_rate, b_size in tqdm(product(models, test_datasets, d_rates, b_sizes),
    #                                                       total=len(models) * len(test_datasets) * len(d_rates) * len(
    #                                                           b_sizes),
    #                                                       desc="Running UQ experiments"):
    #         #print(f"\n 🔍 UQ method: {uq_m} | 🧠 Model: {model_n} | 📊 Dataset: {test_dataset} "
    #               f"| 💧 Dropout rate: {d_rate}| 💧 Block size: {b_size} \n")
    #
    #         uq_main(uq_method=uq_m, uq_config=[d_rate, b_size], test_data=test_dataset, model_version=model_n,
    #                 n_predictions=10)
