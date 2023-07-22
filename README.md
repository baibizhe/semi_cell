## Environments and Requirements

- Windows/Ubuntu both works
- gpu memory>= 8GB required
- CUDA version testted on 11.1
- python=3.8

To install requirements:

```setup
pip install -r requirements.txt
```



## Dataset TODO

- A link to download the data (if publicly available)
- A description of how to prepare the data (e.g., folder structures) 

## Preprocessing TODO to fit 

A brief description of the preprocessing method


Running the data preprocessing code:

```python
python preprocessing.py --input_imgs_path <path_to_input_data> --input_masks_path <path_to_input_data> --output_path <path_to_output_data>
```
Then organize your data dir as follows to fit the training yaml:
data
--images
----train
----val
--labels
----train
----val

## Training

1. To train the model(s) in the paper, run this command:

```bash
python train.py --img 1280 --batch 8 --epochs 50 --data ./cell.yaml --weights yolov5s6.pt --multi-scale --seed 42 --name fold_1
```

You can download trained models here:  TODO




2. To fine-tune the model on a customized dataset, run this command:

```bash
python finetune.py --input-data <path_to_data> --pre_trained_model_path <path to pre-trained model> --other_flags
```




## Inference

1. To infer the testing cases, run this command:

```python
python inference.py --input-data <path_to_data> --model_path <path_to_trained_model> --output_path <path_to_output_data>
```

> Describe how to infer testing cases with the trained models.

2. [Colab](https://colab.research.google.com/) jupyter notebook

3. Docker containers on [DockerHub](https://hub.docker.com/)

```bash
docker container run --gpus "device=0" -m 28G --name algorithm --rm -v $PWD/CellSeg_Test/:/workspace/inputs/ -v $PWD/algorithm_results/:/workspace/outputs/ algorithm:latest /bin/bash -c "sh predict.sh"
```

## Evaluation

To compute the evaluation metrics, run:

```eval
python eval.py --seg_data <path_to_inference_results> --gt_data <path_to_ground_truth>
```

>Describe how to evaluate the inference results and obtain the reported results in the paper.



## Results

Our method achieves the following performance on [Brain Tumor Segmentation (BraTS) Challenge](https://www.med.upenn.edu/cbica/brats2020/)

| Model name       |  DICE  | 95% Hausdorff Distance |
| ---------------- | :----: | :--------------------: |
| My awesome model | 90.68% |         32.71          |

>Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 





### inference

```
python custom_det.py --img 1280 --source patched_cache --weights runs/fold_4.pt runs/fold_3.pt runs/fold_2.pt runs/fold_1.pt runs/fold_0.pt --name testa --max-det 20000 --half --iou-thres 0.5 --conf-thres=0.4 --save-txt --save-conf --line-thickness 1 --hide-labels --project patched_cache/detect --nosave
```
