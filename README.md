# Road Recognition with U-Net

## Impostazioni

Il progetto è strutturato in modo che siano presenti:

1. Directory "Train/" a sua volta suddivisa in:
    - "txt_data/" in cui vengono salvati i file di testo contententi le coordinate delle immagini RGB
    - "json_data/" contentente i file json con le informazioni sulle tracce
    - "mask/" contenente le maschere binarie di groundthrouth
    - "rgb/" contenente le immagini aeree RGB

2. Directory "Test/" a sua volta suddivisa in:
    - "txt_data/" in cui vengono salvati i file di testo contententi le coordinate delle immagini RGB
    - "json_data/" contentente i file json con le informazioni sulle tracce
    - "mask/" contenente le maschere binarie di groundthrouth
    - "rgb/" contenente le immagini aeree RGB

3. Directory "Model/" in cui verranno salvati i modelli addestrati

4. Directory "Prediction/" in cui verranno salvate le predizioni in fase di testing

## Download del dataset

Il download del dataset parte dalla creazione di due file di testo, uno per il train set e uno per il test set, in cui specificare le coordinate mondo
di un rettangolo immmaginario da cui estrarre le immagini rgb.

Successivamente eseguire il comando:

```
python3 -m get_dataset --config <path/train_test/file.txt> --data <dir/salvataggio/train_test_set> --num <num_img_rettangolo>
```


## Addestramento

Per addestrare la rete è sufficiente eseguire il comando:

```
python3 -m train --data <dir/training_set> --model <dir/salvataggio/modello>
```

## Testing

Per eseguire il testing della rete eseguire:

```
python3 -m test --data <dir/testing_set> --pred <dir/salvataggio/predizioni> --model<dir/contenente/modello>
```
