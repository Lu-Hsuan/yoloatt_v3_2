import argparse

class YOLOATT_OPTION:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="YoloAtt options")

        self.parser.add_argument("--model_name", type=str, help="mdoel name", default='yoloatt')
        self.parser.add_argument("--data_path", type=str, help="path to the data", default='../data')
        self.parser.add_argument("--width", type=int, help="width of input", default=320)
        self.parser.add_argument("--height", type=int, help="height of input", default=224)

        self.parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
        self.parser.add_argument("--epochs", type=int, help="train epochs", default=15)
        self.parser.add_argument("--decay_step", type=int, help="period of epoch to apply decay", default=10)
        self.parser.add_argument("--decay_factor", type=float, help="decay factor", default=0.1)
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=8)

        self.parser.add_argument("--no_cuda", help="if set, use cpu as device", action='store_true')
        self.parser.add_argument("--num_workers", type=int, help="number of workers of dataloader", default=8)
        self.parser.add_argument("--save_period", type=int, help="period of saving", default=5)
        self.parser.add_argument("--log_period", type=int, help="period of logging", default=100)
        self.parser.add_argument("--log_path", type=str, help="log path (include weight)", default='log')

        self.parser.add_argument("--weight", type=str, help="path to the weight", default=None)
        self.parser.add_argument("--use_yolov3", help='if set, darknet53 use weight of yolov3', action="store_true")
        self.parser.add_argument("--debug", help='if set, print debug info', action="store_true")
        self.parser.add_argument("--see_grad", help="if set, print mean grad in each parameter", action="store_true")
        # for object detection
        self.parser.add_argument("--loss_weight", type=float, help="weight of obj", default=0.1)

        self.parser.add_argument("--obj_names", type=str, default="./coco.names", help="path to obj names")
        self.parser.add_argument("--obj_path", type=str, help="path to the obj", default='./')
        self.parser.add_argument("--img_size", type=int, default=416, help="size of image for object detection")
        self.parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options