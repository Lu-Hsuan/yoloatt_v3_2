# Saliency-object-Detection
  
## yoloatt_saliency
  
### Train yoloatt
    python yoloatt_train.py --data_path=data_path --weight=weight_path --log_path=output_path --batch_size=batch_size --epochs=epochs
  
### Evaluate yoloatt saliency map metrics
   python yoloatt_eval.py --data_path=data_path --weight=weight_path --log_path=output_path
   
### Predict and save img on yoloatt 
    python yoloatt_test.py --data_path=data_path --weight=weight_path --log_path=output_path
    
### Predict and save img on yoloatt 
    python yoloatt_test_save.py --data_path=data_path --weight=weight_path --log_path=output_path
  
### Code memo
  * TODO
  * TODO
### Data & weight
[Goole drive](https://drive.google.com/drive/folders/1s-xrGMb26etWnLVvbrngF0eKq3FvAbth?usp=sharing)    
    Please change your path    
    init path is : './data'
  
## Saliency Object Example
---
![method5_Salobj](https://github.com/Lu-Hsuan/yoloatt_v3_2/blob/master/img_example/sal_obj_test%203_ch.png)
  
