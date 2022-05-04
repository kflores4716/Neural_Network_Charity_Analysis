# Neural_Network_Charity_Analysis

## Overview 

In this analysis, we are working with neural networks and using a CSV that contains information on applicants who are looking to receive funding from Alphabet Soup. We created a deep learning binary classifier model that can predict whether the applicants will be successful or not based on the features in the dataset. This way Alphabet Soup can better determine who to provide funding for.  With so many organizations in need of money, it can be risky deciding which ones to provide it to, so we are creating a model that will help guide Alphabet Soup through that decision-making process. 

We used `Jupyter Notebook` to clean, train, and test the data, using the `TensorFlow Library` for the latter two processes. 


## Results 

### Data Preprocessing 

##### Target(s) 

- The target variable for this model was the `IS_SUCCESSFUL` column. 

##### Feature(s) 

- All other columns in the dataset were considered features for this model, although some of the columns were determined to not be needed and were removed from the dataset altogether. 

- Those columns that were removed were the `EIN` and `NAME` columns. 

### Compiling, Training, and Evaluating the Model 

- Initially, we used two layers in our model, the first layer containing 80 neurons and the second layer containing 30. We started with 80 neurons because in general, we want to have twice as many neurons as input features, and after preprocessing our data, we had about 40 input features. For our activation function, we used `relu` on both input layers because of its simplifying output. We then used the `sigmoid` function for our output layer because it transforms the output to a range between 1 and 0, which is what we are looking for in this model. 

- With this initial draft of our model, we didn’t quite reach our target model performance of 75%, as the model resulted in just over 72% performance. Because the model wasn’t achieving our desired performance, we began making changes to the model in an attempt to increase its performance. 

- First, we decided to remove the `SPECIAL_CONSIDERATIONS_N` column from our features, as special considerations are only taken into account if they exist. It doesn’t make much sense to have a column of data that won’t be considered in our analysis. Next, we added some neurons to the model, as well as an additional hidden layer to see if that could increase model accuracy. Lastly, we changed the `relu` activation function to `softmax` because `softmax` tends to be used for probability distributions, which is essentially what we are doing in this analysis. While we ran our model after each change we made, we unfortunately were still unable to boost its performance to exceed our 75% target.  


## Summary 

Although we tried adjusting our model to perform better, we were unable to get our neural network to reach our goal of 75% accuracy. The changes we made never seemed to detract from the deep learning model’s accuracy, but they didn’t cause much of an improvement either. In fact, they only increased the accuracy of our model from 72.47% to 72.59%, which is so marginal that these changes can be considered unnecessary. Due to this lack of improvement, we could probably stand to remove the extra layer and nodes that we added, as they might be slowing down the model without adding value.  

Alternatively, we could try to implement a different machine learning model to see if it better predicts the success of these applicants. Perhaps a `Support Vector Machine` supervised learning model would be able to predict success more accurately. SVM models are often used for binary data classification, and that is exactly what we are doing in this analysis. An SVM model could consider several of the features in this data set while also being able to handle any outliers that may exist. We could very well run the model and find that it is no better at predicting the success of our applicants; however, it wouldn’t take very long to set up and would definitely be worth trying.  
