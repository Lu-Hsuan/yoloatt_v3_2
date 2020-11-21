# Saliency-Detection
  
## yoloatt_saliency
  
### Train yoloatt
    python yoloatt_train.py --data_path=data_path --weight=weight_path --log_path=output_path --batch_size=batch_size --epochs=epochs
  
### Evaluate yoloatt saliency map metrics
    python yoloatt_eval.py --data_path=data_path --weight=weight_path --log_path=output_path
   
### Predict and save npy on yoloatt 
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
  
## Saliency  Example
---
### data preprocess
![mean_std](https://github.com/Lu-Hsuan/yoloatt_v3_2/blob/master/%40yoloatt_saliency/!ex_img/mean_std_2_f.png)
---
### Predict example
![example](https://github.com/Lu-Hsuan/yoloatt_v3_2/blob/master/%40yoloatt_saliency/!ex_img/mean_std_example_nomean_2.png)
  
