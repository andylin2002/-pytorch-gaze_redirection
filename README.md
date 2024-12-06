## Dependencies

Python == 3.10.15 <br />
torch == 2.5.1 <br />
dlib == 19.24.6

## Train

```Bash
python main.py --mode train --data_path /home/andy/AILab/AIfinal/pytorch_gaze_redirection-master/eyespatch_dataset/all --log_dir ./log/ --vgg_path ./vgg16_reducedfc.pth
```

## Test
```
python main.py --mode eval --client_pictures_dir ./client_pictures/ --log_dir ./log/
```

## VGG16 pretrained weights
```Bash
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

## Push
```
git push -u origin
```
### if "rejected because the remote contains work that you do not have locally."
```
git pull --rebase
```
