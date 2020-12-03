import os
import torch
from torchsummaryX import summary
import time
from models_rect import Darknet
def time_cal2(m):
    times = []
    batch = 16
    with torch.no_grad():   
        m.eval()
        x_i = torch.rand(batch,3,224,320).to('cuda')
        m.forward(x_i)
        for i in range(100):
            x_i = torch.rand(batch,3,224,320).to('cuda')
            torch.cuda.synchronize()
            t0 = time.time()
            m.forward(x_i)
            torch.cuda.synchronize()
            dt = time.time() - t0
            times.append(dt)
    dt = sum(times) / len(times)
    print(f"Avg {batch}-frame time: {dt:.4f} s ({1 / dt:.1f} fps)")
    print(f"Avg single-frame time: {dt/batch:.4f} s ({batch / dt:.1f} fps)")
    return dt

if __name__ == "__main__":
    cfg_path = "yoloatt_v3_split_rect.cfg"
    print(cfg_path)
    model = Darknet(cfg_path).to('cuda')
    summary(model,torch.rand(1,3,224,320).to('cuda'))
    print('Model cfg: ',cfg_path)
    time_cal2(model)