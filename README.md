# A-Hierarchical-Deep-Temporal-Model-for-Group-Activity-Recognition
Paper implementation for Hierarchical Deep Temporal Model for Group Activity Recognition, computer vision


Paper link : [LINK](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)

Dataset link (Google drive) :  [LINK](https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS?usp=sharing)


## Abstract
In group activity recognition, the temporal dynamics of the whole activity can be inferred based on the dynamics of the individual people representing the activity. We build a deep model to capture these dynamics based on LSTM models. To make use of these observations, we present a 2-stage deep temporal model for the group activity recognition problem. In our model, a LSTM model is designed to represent action dynamics of individual people in a sequence and another LSTM model is designed to aggregate person-level information for whole activity understanding. We evaluate our model over the volleyball dataset.


Model

![image](https://github.com/user-attachments/assets/41a8541a-793a-481e-b25d-2fe84344ecbe)

Figure 1

Figure 1: High level figure for group activity recognition via a hierarchical model. Each person in a scene is modeled using a temporal model that captures his/her dynamics, these models are integrated into a higher-level model that captures scene-level activity.


![image](https://github.com/user-attachments/assets/a7c5c12b-0006-470d-9607-61a2a9104e74)

Figure 2: Detailed figure for the model. Given tracklets of K-players, we feed each tracklet in a CNN, followed by a person LSTM layer to represent each player's action. We then pool over all people's temporal features in the scene. The output of the pooling layer is feed to the second LSTM network to identify the whole teams activity.


The train-test split of is performed at video level, rather than the frame level, to evaluate models more convincingly. The list of action and activity labels and related statistics are tabulated in the following tables:

| Group Activity Class | No. of Instances |
| ------------- | ------------- |
|Right set	|644|
|Right spike|	623|
|Right pass|	801|
|Right winpoint|	295|
|Left winpoint|	367|
|Left pass|	826|
|Left spike|	642|
|Left set|	633|


|Action Classes|	No. of Instances|
| ------------- | ------------- |
|Waiting|	3601|
|Setting	|1332|
|Digging	|2333|
|Falling	|1241|
|Spiking	|1216|
|Blocking|	2458|
|Jumping	|341|
|Moving	|5121|
|Standing|38696|


Further information:

The dataset contains 55 videos. Each video has a folder for it with unique IDs (0, 1...54)

Train Videos: 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54

Validation Videos: 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51

Test Videos: 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47



<img width="350" alt="image" src="https://github.com/user-attachments/assets/0805b0ed-2dda-4d17-af03-fea5977743aa">

According to the original paper, the models are as the following



|My Model|	Paper model|
| ------------- | ------------- |
|B1|	B1|
|B2	|B3|
|B3	|B4|
|B4	|B5|
|B5 |B6|
|B6	|TWO STAGE MODEL WITH ONE GROUP|
|B7	|TWO STAGE MODEL WITH TWO GROUPS|


You can find a presentation for the paper [here](https://docs.google.com/presentation/d/1iHMRCghn-dOYc2knvTj8Kp27RRojCsLzCbE8Ax5JCOs/edit#slide=id.p4)


