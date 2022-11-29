import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import numpy as np
import cv2
from glob import glob
from keras.models import load_model
from keras.metrics import MeanIoU
from tqdm import tqdm

IMG_PATH = 'rgb/*'
MASK_PATH = 'mask/'

SIZE = 512

def iou_metric(y_true_in, y_pred_in):
    labels = y_true_in
    y_pred = y_pred_in
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
    intersection = temp1[0]
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9
    iou = intersection / union
    return iou[0][0]

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data', help='Path to the testing set directory.', required=True)
    parser.add_argument('--pred', help='directory for saving the predicted masks.', required=True)
    parser.add_argument('--model', help='Directory for loading the model.', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # TEST
    test_images = glob(os.path.join(args.data, IMG_PATH)) # load the test images
    model = load_model(args.model) # load the model
    miou = MeanIoU(num_classes=2)
    miou_tmp = MeanIoU(num_classes=2)

    iou_list = []
    triple_list = []
    auto = True
    TH = 0.25
    SAVE = True

    for path in tqdm(test_images, total=len(test_images)):
        # read and preprocess test image
        test_img = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = test_img.shape
        test_img = cv2.resize(test_img, (SIZE, SIZE))
        test_img = test_img / 255.0
        test_img = test_img.astype(np.float32)
        test_img_exp = np.expand_dims(test_img, axis=0) # (256,256,3) -> (1,256,256,3), images as batch
        pred_mask = model.predict(test_img_exp)[0] # predict mask for test image

        # convert mask from grayscale to binary, 2 ways of thresholding
        if not auto:
            pred_mask[pred_mask >= TH] = 1
            pred_mask[pred_mask < TH] = 0
    
        # convert mask from float to uint8
        pred_mask *= 255
        pred_mask = pred_mask.astype(np.uint8)
        
        if auto:
            _, pred_mask = cv2.threshold(pred_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        pred_mask = cv2.resize(pred_mask, (w, h)) # resize mask to original size
        
        # open ground truth mask
        name = path.split("/")[-1]
        t_path = os.path.join(args.data, MASK_PATH, name)
        truth = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)

        # compute miou
        p = pred_mask / 255.0
        p = p.astype(np.float32)
        t = truth / 255.0
        t = t.astype(np.float32)
        
        miou_tmp.reset_state()
        miou_tmp.update_state(p, t)
        miou.update_state(p, t)
        custom = iou_metric(t, p)
        iou_list.append(custom)

        triple_tmp = (name, miou_tmp.result().numpy(), custom)
        triple_list.append(triple_tmp)

        # sometimes saving output imgs is not needed
        if SAVE:
            # create output image as: rgb | prediced mask | true mask
            roads = np.argwhere(truth == 255) # get position of white pixels (true roads)
            # convert masks to color imgs
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
            truth = cv2.cvtColor(truth, cv2.COLOR_GRAY2RGB)
            # change the true roads color to red
            for coord in roads:
                truth[coord[0], coord[1]] = [0,0,255]
            # create and save the output img
            test_img = cv2.resize(test_img, (w, h))
            final = np.hstack((test_img, pred_mask, truth))
            cv2.imwrite(os.path.join(args.pred, name), final)

    final_miou= miou.result().numpy()
    custom_iou = sum(iou_list)/len(iou_list)
    
    if auto:
        print('\n\notsu\t' + str(final_miou) + '\t' + str(custom_iou))
    else:
        print('\n\n' + str(TH) + '\t' + str(final_miou) + '\t' + str(custom_iou))

if __name__ == "__main__":
    main()