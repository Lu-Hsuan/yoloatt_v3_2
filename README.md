# yoloatt_v3_2
---
## Train yoloatt with frozen yolov3 part
    python -u yoloatt_train_frozen.py --epochs=epochs --batch_size=batch_size --log_period=log_period --save_period=1 --log_path=log_path --weight=yolov3_weight.pth --lr=lr
  
## Evaluate yoloatt saliency map metrics
    python -u yoloatt_eval.py --batch_size=32  --log_path=log_path --weight=weight.pth
  
## Predict and save validation img on yoloatt 
    python -u dect_obj_att_3_pred_split_mix.py --out_path=out_path --weight=weight.pth --batch_size=batch_size
  
## Predict and save test img on yoloatt 
    python -u dect_obj_att_3_pred_split_test.py --data_path=img_data_path --out_path=img_save_path --weight=weight.pth --batch_size=batch_size

## Saliency Object Example
---
![method5_Salobj](https://github.com/Lu-Hsuan/yoloatt_v3_2/blob/master/img_example/sal_obj_test%203_ch.png)
  
