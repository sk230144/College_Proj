﻿# Brain_Tumor_Detection
Technologies :- Bootstrap , CSS , Javascript and Deep Learning ( CNN )
In brain tumor detection, there are two types of image folders: one is "yes," which stores images 
containing a brain tumor, and the other is "no," which stores images that do not have a brain tumor. The 
Python file trains the model using images from both the "yes" and "no" folders. When testing an uploaded 
image, it checks whether the image contains a brain tumor. If a brain tumor is detected, it indicates that 
the image shows the presence of a brain tumor; otherwise, it indicates that the image does not have a 
brain tumor.

![Screenshot (402)](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/18dfe479-8e9b-4085-9582-e73986fd78d8)
![Screenshot (403)](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/e6d55809-0134-438a-9de5-9613f0d26da1)
![Screenshot (404)](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/2b9666e3-6e67-4399-abf7-9dd53f497740)

# Abstract
A Brain tumor is considered as one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors. Every year, around 11,700 people are diagnosed with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately 34 percent for men and36 percent for women. Brain Tumors are classified as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment, planning, and accurate diagnostics should be implemented to improve the life expectancy of the patients. The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data is generated through the scans. These images are examined by the radiologist. A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties.
Application of automated classification techniques using Machine Learning(ML) and Artificial Intelligence(AI)has consistently shown higher accuracy than manual classification. Hence, proposing a system performing detection and classification by using Deep Learning Algorithms using Convolution-Neural Network (CNN), Artificial Neural Network (ANN), and Transfer-Learning (TL) would be helpful to doctors all around the world.

# Context
Brain Tumors are complex. There are a lot of abnormalities in the sizes and location of the brain tumor(s). This makes it really difficult for complete understanding of the nature of the tumor. Also, a professional Neurosurgeon is required for MRI analysis. Often times in developing countries the lack of skillful doctors and lack of knowledge about tumors makes it really challenging and time-consuming to generate reports from MRI’. So an automated system on Cloud can solve this problem.

# Definition
To Detect and Classify Brain Tumor using, CNN and TL; as an asset of Deep Learning and to examine the tumor position(segmentation).

# About the data:
The dataset contains 3 folders: yes, no and pred which contains 3060 Brain MRI Images.

# Folder	Description
Yes	The folder yes contains 1500 Brain MRI Images that are tumorous
No	The folder no contains 1500 Brain MRI Images that are non-tumorous
