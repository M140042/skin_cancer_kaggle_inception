# Introduction
This is a five day’s project for Kaggle Competition: **Skin Cancer MNIST: HAM10000**. 

**Skin cancer** is a common disease that affect a big amount of peoples and therefore early detection is critical. 

This is an **image classification** problem, which required me to classify 7 different classes of skin cancer 
which are listed below:
1. Melanocytic nevi 
2. Melanoma 
3. Benign keratosis-like lesions
4. Basal cell carcinoma 
5. Actinic keratoses 
6. Vascular lesions 
7. Dermatofibroma


# Methodology

_2.1. Dataset_

Dataset contain 10,015 images. I split the dataset into **80%** training data and **20%** test data.

_2.2 Observations from EDA_

1.	Huge **class imbalance**, cell type **Melanecytic nevi** is the dominant class.

2.	Larger instances of patients having age from 30 to 60.

_2.3 Model_

Due to resource I have, I applied **transfer learning** method to speed up my training process. 

**Inception v3** is a widely-used image recognition model that has been shown to attain greater than 78.1% accuracy on the ImageNet dataset. Therefore, inception V3 is picked as my main model.

_2.4 Implementation_

I had trained two models in total. 

Only images have been used to train my model. 

**Frist model (Inception v3 1st model):**

1.	Freeze all the layers in Inception, then connect the last layer with a Global Spatial Average Pooling layer. 
Reason for using this layer is based on <a href='https://arxiv.org/pdf/1312.4400.pdf'>this paper</a> and also this <a href='https://www.quora.com/What-is-global-average-pooling'>link</a> as well. 

2.	Add a Softmax layer for 7 classes of skin cancer as output layer.

3.	Train it for five epochs and then fine tune process by selecting and training the top 2 inception 
blocks (all remaining layers after 249 layers in the combined model) for ten epochs.

4.	Optimizer: Rmsprop with 0.001 learning rate, 0.9 rho, 0.1 epsilon and 0.9 decay.

**Second model (Inception v3 2nd model):**

1.	Freeze all layers except for layers that have moving mean and variance so that 
weights will be adjusted to the mean/variance of the new dataset.

2.	Then, connect with Global Max Pooling Layer, followed by one dense layer with 
Rectify Linear Unit (ReLu) as activation function, dropout layer and one softmax layer as output.

3.	Train for 3 epochs and fine tune process by selecting and training the top 2 inception 
blocks (all remaining layers after 249 layers in the combined model) for ten epochs.

4.	Optimizer: Adam with learning rate 0.0001, beta_1 0.9, beta_2 0.999, decay 0.

5.	Refer to this <a href='https://github.com/hoang-ho/Skin_Lesions_Classification_DCNNs/blob/master/Fine_Tuning_InceptionV3.ipynb?fbclid=IwAR0ZLphprXQe2kJmy_OMAxOgIIZMmomubbSSQYD8B9wyRZaGBMsL5DHg8QU'>link</a> for detailed implementation.

3. Results

# Model
| Training accuracy | Validation Accuracy | F1 score |
| ----------------- | ------------------- | -------- |
| Inception v3 1st model | 73.81% | 66.05% | 0.116 |
| Inception v3 2nd model | 70.69% | 72.18% | 0.128 |

From the above table, Inception v3 2nd model clearly performs 
better in terms of validation accuracy and f1 score. In fact, 
I could train more epochs to increase the performance as the 
validation accuracy is still higher than training accuracy. 
However, due to time constraint and limited computing resources, 
I only managed to train a total of 13 epochs.

# Further Improvements

1.	Train more epochs for Inception v3 2nd model as validation accuracy is still higher than the training accuracy.

2.	Try out other transfer learning models such as VGG19, Resnet50 and etc.

3.	Using SMOTE or other oversampling method to increase f1-score.

4.	Performs more pre-processing technique on training data.

5.	Cross-validation can be done to find the optimal hyper-parameters.
