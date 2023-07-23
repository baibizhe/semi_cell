## Environments and Requirements

- Windows/Ubuntu both works
- gpu memory>= 8GB required
- CUDA version testted on 11.1
- python=3.8

To install requirements:

```setup
pip install -r requirements.txt
```
# TODO
DATASET

## Dataset 

- Download data from https://neurips22-cellseg.grand-challenge.org/dataset/
-  folder structures  will be describe in preprocessing section

## Preprocessing


Running the data preprocessing code:

```python
python create_annotation_for_yolo.py --input_imgs_path <path_to_input_data> --input_masks_path <path_to_input_data> --output_path <path_to_output_data>
```
Then organize your data dir as follows to fit the training yaml:

data \
--images \
----train \
----val \
--labels \
----train \
----val 

## Training

1. To train the model(s) in the paper, run this command:
cell.yaml is provided
download yolov5s6.pt from https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```bash
python train.py --img 1280 --batch 8 --epochs 50 --data ./cell.yaml --weights yolov5s6.pt --multi-scale --seed 42 --name fold_1
```

You can download trained models here:  
https://drive.google.com/file/d/1ty6jXk-ZnuA4f23ZvlLTvRdWhVY8-JM3/view?usp=drive_link

https://drive.google.com/file/d/1p42Wm1zEHprFeDW8KxgYJa7PYsGnL4GZ/view?usp=drive_link

https://drive.google.com/file/d/1NkbHfEZWdMAZBR8toWk4Nvf_c3srd-Cn/view?usp=drive_link

https://drive.google.com/file/d/1n7o7dvIygyI-JjECKRwW2hkl4puitWvp/view?usp=drive_link

https://drive.google.com/file/d/1D2Ynk9loeJZHAKjS351na7f-6wrIetjh/view?usp=drive_link


2. To fine-tune the model on a customized dataset, run this command:
first preprocessing data,
```python
python create_annotation_for_yolo.py --input_imgs_path <path_to_input_data> --input_masks_path <path_to_input_data> --output_path <path_to_output_data>
```
then train the model
```bash
python train.py --img 1280 --batch 8 --epochs 50 --data ./youdata.yaml --weights yolov5s6.pt --multi-scale --seed 42 --name your_run_name
```




## Inference  

1. To infer the testing cases, run this command:

```python
python custom_det.py --img 1280 --source patched_cache --weights runs/fold_4.pt runs/fold_3.pt runs/fold_2.pt runs/fold_1.pt runs/fold_0.pt --name testa --max-det 20000 --half --iou-thres 0.5 --conf-thres=0.4 --save-txt --save-conf --line-thickness 1 --hide-labels --project patched_cache/detect --nosave
```


```bash
docker container run --gpus "device=0" -m 28G --name algorithm --rm -v $PWD/CellSeg_Test/:/workspace/inputs/ -v $PWD/algorithm_results/:/workspace/outputs/ algorithm:latest /bin/bash -c "sh predict.sh"
```



2. [Colab]([https://colab.research.google.com/](https://colab.research.google.com/drive/1hrsotfDiyyWpdlWIY2JpkzNLjURE5joG?usp=sharing)) jupyter notebook

3. Docker containers on [DockerHub]([https://hub.docker.com](https://hub.docker.com/r/baibizhe/overoverfitting)/)



