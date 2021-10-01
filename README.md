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
- Easiest to train.
- Fewer epochs it performance very good.
- Lightweight models.
- MAP and FPS both are good.
 ![Screenshot (41)](https://user-images.githubusercontent.com/47352327/135662100-4bac9d81-b2c0-4b49-b0dd-8bdce1becc72.png)

And Yolov5 has various types but I have selected Yolov5m because
- MAP and FPS both are good.
- Lightweight model.
 ![Screenshot (59)](https://user-images.githubusercontent.com/47352327/135662664-8dc59f7b-fc13-41b0-8b60-67717b9921cc.png)
 
2.**Prepare Dataset for training:** I have 4750 annotated images. And that annotation is XML format but we need txt format for Yolov5 model training. That`s why I had to change XML annotation to txt annotation using XmlToTxt (https://github.com/Isabek/XmlToTxt).
Then I split the dataset into two-part –
- Train data : 3811
- Valid data : 939
3.	**Model Training:**
I trained the Yolov5 model on Google Colab.
4.	**Model Evaluation:**
-	MAP@0.5 –  62.4%
-	FPS – 0.35 (CPU i5 3rd GEN)

- **Confusion Matrix:**
 ![confusion_matrix](https://user-images.githubusercontent.com/47352327/135662934-aaded062-fd4a-4297-8c70-0873cc9ce079.png)

- **Results:**
![results](https://user-images.githubusercontent.com/47352327/135663181-4b4c9fdd-f2ff-4275-910f-4ba9c7868286.png)

- **Output of test images:**
  ![test_batch1_labels](https://user-images.githubusercontent.com/47352327/135663304-5d7588b5-43ba-4020-9748-2c024201b8f3.jpg)
                                                 
### Project demo:
 ![HardHat_output (online-video-cutter com) (1)](https://user-images.githubusercontent.com/47352327/135667045-07f513a2-624f-43b8-b98f-aa40d8edf63e.gif)




