# Safety Equipment Detection System
  

### Description: 
Build a Personal Safety Equipment Detection System.

### Technical Specification: 
Toward the solution of Build a Personal Safety Equipment Detection System using below technical required.
1.	Programming language  -	Python
2.	IDE	                   - VS code
3.	Cloud	                - Google colab
4.	Object detection Model -	Yolov5
5.	Image processing -	OpenCV
6.	Deep learning Library - Pytorch


### Training Process: 
Training process I have followed these steps.
1.	**Model selection:** we have many object detection model like: Yolov3, Yolov4, Yolov5 and EfficientDet models etc.
But I have selected Yolov5 because –
•	Easiest to train.
•	Fewer epochs it performance very good.
•	Lightweight models.
•	MAP and FPS both are good.
 
       



	

And Yolov5 has various types but I have selected Yolov5m because
o	MAP and FPS both are good.
o	Lightweight model.
 

2.	**Prepare Dataset for training:** I have 4750 annotated images. And that annotation is XML format but we need txt format for Yolov5 model training. That`s why I had to change XML annotation to txt annotation using XmlToTxt (https://github.com/Isabek/XmlToTxt).
Then I split the dataset into two-part –
- Train data : 3811
- Valid data : 939
3.	**Model Training:**
I trained the Yolov5 model on Google Colab.
4.	**Model Evaluation:**
-	MAP@0.5 –  62.4%
-	FPS – 0.35 (CPU i5 3rd GEN)

- **Confusion Matrix:**
 



- **Results:**
 
- **Output of test images:**
                                                    
### Project demo:
 



