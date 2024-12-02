## Dependencies

tensorflow == 2.18.0 <br />
Python == 3.10.15


## Train

```Bash
python main.py --mode train --data_path /home/andy/AILab/AIfinal/pytorch_gaze_redirection-master/eyespatch_dataset/all --log_dir ./log/ --vgg_path ./vgg_16.ckpt
```

## VGG16 pretrain weight
```Bash
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```
