# Road Recognition with U-Net

## Setting Up

This project needs some folders:

1. Directory "Train/" divided into:
    - "txt_data/" in which store the coordinates of the RGB images;
    - "json_data/" in which store road informations;
    - "mask/" in which store the groundthrouth binary mask;
    - "rgb/" in which store the RGB images.

2. Directory "Test/" divided as before;

3. Directory "Model/" where store the generated models;

4. Directory "Prediction/" where save the testing predictions;

## Dataset Download

To obtain the dataset, it's required to create a txt file for the training set and one for the test set. Both have to contain the coordinates of a "rectangle" from which extract the aerial RGB images.

The executes:

```
python3 -m get_dataset --config <path/to/train_test/file.txt> --data <path/to/train_test_set> --num <tot_num_images>
```


## Training

To train the network, just execute:

```
python3 -m train --data <path/to/training_set> --model <path/to/model/folder>
```

## Testing

To test the network, just execute:

```
python3 -m test --data <path/to/testing_set> --pred <path/to/prediction/folder> --model<path/to/model>
```
