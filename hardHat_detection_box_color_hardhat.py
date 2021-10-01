import cv2
import torch
import numpy as np
import datetime
# from PIL import Image
# import argparse
# import time
# from pathlib import Path
from yolov5.utils.datasets import letterbox
import cv2
import torch
import random
# from numpy import random
# from yolov5.models.experimental import attempt_load
# from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size,non_max_suppression,scale_coords,set_logging
# from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device,time_synchronized

class HardHatDetector:
    def __init__(self,model_path,video_capture=None,output_video="output.mp4"):
        self.model_path = model_path
        self.video_capture = video_capture
        self.output_video = output_video
        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.out = cv2.VideoWriter(self.output_video,self.fourcc, 5, (1280,720))

    def bbox_rel(self,*xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    def calculate_fps(self,start_time,framec):
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = framec / elapsed_time
        return fps

    def draw_bboxes(self,img,label_c,box,tl,color,fps):
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label_c, 0, fontScale=tl / 3, thickness=tf)[0]
        t_color = (255,255,255)
        if color[0]>225 and color[1]>225 and color[2]>225:
            t_color=(255,0,0)
        cv2.rectangle(img, c1, c2, color,tl,cv2.LINE_AA)
        if c1[1]>10:
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label_c, (c1[0], c1[1] - 2), 0, tl / 3, t_color, thickness=2, lineType=cv2.LINE_AA)
        else:
            c1 = c2[0] - t_size[0], c2[1] + t_size[1] + 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label_c, (c1[0], c1[1]-2), 0, tl / 3, t_color, thickness=2, lineType=cv2.LINE_AA)

        # cv2.putText(img, label_c, (c2[0], c2[1] + 2), 0, tl / 3, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        # cv2.putText(img, f"fps:{fps:.2f}", (550, 30), 0, tl / 3, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        return img

    def load_model(self,dev="cpu"):
        # Initialize
        set_logging()
        device = select_device(dev)
        # Load model
        # model = attempt_load("yolov5_model/yolov5x.pt", map_location=device)  # load FP32 model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=self.model_path)
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(640, s=stride)  # check img_size
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        return model,stride,device,imgsz

    def detect_boxes_classes(self,img,model,stride,device,imgsz):
        # names = model.module.names if hasattr(model, 'module') else model.names
        names = ["hardhat",'head',"person"]
        # t0 = time.time()
        im0s = img.copy()
        img = letterbox(im0s, imgsz, stride=stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        half = device.type != "cpu"  # half precision only supported on CUDA
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            # Inference
        # t1 = time_synchronized()
        pred = model(img, augment=True)[0]
        # print(pred)
        # Apply NMS
        pred = non_max_suppression(pred, 0.50, 0.5, classes=[0,1], agnostic=True)
        t2 = time_synchronized()
        xywhs,labels,xyxys,confs = [],[],[],[]
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                     label = f'{names[int(cls)]}'
                     xywh = self.bbox_rel(*xyxy)
                     xyxys.append(xyxy)
                     xywhs.append(xywh)
                     labels.append(label)
                     confs.append([conf.item()])
                # print(labels)
        return xyxys,xywhs,labels,confs,im0

    def detect_on_video(self):
        num_frame = 0
        start_time = datetime.datetime.now()
        while True:
            ret, frame = self.video_capture.read()
            num_frame += 1
            if ret != True:
                print("wrong source path")
                break
            xyxys,xywhs,labels,confs,img = self.detect_boxes_classes(frame,model,stride,device,imgsz)
            fps = self.calculate_fps(start_time,num_frame)
            print(fps)
            print(labels)
            for i,(box,label) in enumerate(zip(xyxys,labels)):
                box = box[:4]
                
                #centroid of box
                x_c = box[0]+(box[2]-box[0])/2
                y_c = box[1]+(box[3]-box[1])/2
                # #color selection
                im = img.copy()
                b,g,r = im[int(y_c)-12,int(x_c)]
                color = (int(b),int(g),int(r))

                label_c = f"{label} {confs[i][0]*100:.0f}%"
                 # cv2.circle(img,(int(x_c),int(y_c)-12),25,color,3)
                img = self.draw_bboxes(img, label_c,box, 2, color,fps)
            #save video
            self.out.write(img)
            cv2.imshow("",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_capture.release()
        self.out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video = "hardhat.mp4"
    model_path = 'hardhat_model/exp4/weights/best.pt'
    output_video = "output/HardHat_output_box_color_hardhat.mp4"
    video_capture = cv2.VideoCapture(input_video)
    detector = HardHatDetector(model_path,video_capture,output_video)
    model,stride,device,imgsz = detector.load_model()
    detector.detect_on_video()