![Machine Learning](Machine.jpg)

# Machine Learning 
## Types of Machine Learning
- There  are Two Types 
    1. Supervised
        - Linear Regression
        - Logistic Regression
        - Decision Trees
        - Random Forest
        - Support Vector Machines (SVM)
        -  K-Nearest Neighbors (KNN)
        - Naive Bayes
        - Neural Networks
        - Gradient Boosting Algorithms (e.g., XGBoost)
        -  Linear Discriminant Analysis (LDA)
    2. Unsupervised
       - K-Means
       -  Hierarchical Clustering
       - Principal Component Analysis (PCA)
       - t-Distributed Stochastic Neighbor Embedding (t-SNE)
       - Apriori Algorithm
       - Generative Adversarial Networks (GANs)
        - Isolation Forest

# Introduction 

## Supervised Machine Learning

### What is it?
     
     Supervised machine learning is a type of machine learning where the algorithm is trained on a labeled dataset, which means that the input data used for training is paired with corresponding output labels. The goal of supervised learning is to learn a mapping or relationship between the input features and the target labels so that the algorithm can make accurate predictions or classifications on new, unseen data.

### Urdu me Dkhy: 
     Machine learning mein supervised learning ek aisi technique hai jahan algorithm ko labeled dataset par train kiya jata hai. Iska matlab hai ki training ke liye istemal kiya gaya input data saath mein corresponding output labels ke saath juda hota hai. Iska uddeshya hai ki algorithm input features aur target labels ke beech ka mapping ya relationship seekhe, taki woh naye, anjaane data par sahi predictions ya classifications kar sake.    

- **Key Aspects**:

    1. Labeled Data:
        - In supervised learning, training data consists of input samples and their corresponding output labels. These labels serve as the ground truth, providing the algorithm with the correct answers during the training phase.
      
    2. Training Algorithm:
        - A specific learning algorithm is chosen based on the nature of the problem (classification or regression) and the characteristics of the data. Common algorithms include linear regression, decision trees, support vector machines, neural networks, and more.

    3. Prediction/Inference:
        - Once trained, the model can be deployed to make predictions or classifications on new, unseen data. This is the practical application of the supervised learning model.

    4. Evalution:    
        - Different evaluation metrics are used depending on the nature of the problem. For classification tasks, metrics like accuracy, precision, recall, and F1 score are common. Regression tasks may use metrics such as mean squared error (MSE) or mean absolute error (MAE).

- **Types**
    - **Classification**:
        In classification, the algorithm learns to categorize input data into predefined classes or labels.
        Example: Spam or not spam, image recognition (identifying objects in an image).



    - **Regression**:
        Regression involves predicting a continuous output variable based on input features.
        Example: Predicting house prices based on features like size, location, and number of bedrooms.

- **Common Algorithm**
   - Certainly! Here are the names of some common supervised learning algorithms:

1. **Linear Regression:**
- Used For Regression Problem
2. **Logistic Regression:**
- Used for classification problems,especially binary problem
3. **Decision Trees :** 
-  Makes decisions by recursively splitting data into subsets.
- Can be use for both classification and regression
4. **Random Forest:**
- Ensemble method using multiple decision trees for better accuracy.
- An Ensamble method can be used for both classification and Regression
5. **Support Vector Machines (SVM):**
- Finds a hyperplane to separate classes in high-dimensional space.
- Primarily used for Classification
6. **K-Nearest Neighbors (KNN):**
- Classifies data based on the majority class of its nearest neighbors.
7. **Naive Bayes:**
- Probabilistic method based on Bayes' theorem for classification.
8. **XGBoost:**
- Gradient boosting algorithm for improved accuracy and performance.
9. **LightGBM:**
- Used For both Regression and classification method
- Gradient boosting framework, similar to XGBoost, optimized for speed.

## Applications:
   Supervised learning has a wide range of applications across various domains. Here are some brief examples:

1. **Image Recognition:**
   - Classifying and recognizing objects or patterns within images, such as facial recognition or identifying objects in photos.

2. **Speech Recognition:**
   - Converting spoken language into text, commonly used in virtual assistants and voice-controlled systems.

3. **Text Classification:**
   - Categorizing text data into predefined categories, like spam detection in emails or sentiment analysis in social media.

4. **Predictive Analytics:**
   - Predicting future outcomes based on historical data, such as stock price prediction or weather forecasting.

5. **Credit Scoring:**
   - Assessing the creditworthiness of individuals based on various financial and personal factors.

6. **Medical Diagnosis:**
   - Predicting disease outcomes or diagnosing medical conditions based on patient data, lab results, and medical history.

7. **Recommendation Systems:**
   - Recommending products, movies, or content based on user preferences and behavior.

8. **Autonomous Vehicles:**
   - Training models to recognize and respond to objects and situations for autonomous driving.

9. **Fraud Detection:**
   - Identifying fraudulent activities in financial transactions or online activities.

10. **Language Translation:**
    - Translating text or speech from one language to another using machine learning models.

11. **Predictive Maintenance:**
    - Anticipating when equipment or machinery is likely to fail, allowing for preventive maintenance.

12. **Biometric Authentication:**
    - Verifying and authenticating individuals based on unique biological characteristics, such as fingerprints or facial features.

Supervised learning is versatile and applicable to a broad range of real-world problems, making it a fundamental and widely used approach in machine learning.

## Challenges:
   While supervised learning has proven to be a powerful and versatile approach, it comes with its set of challenges. Some common challenges include:

1. **Insufficient Data:**
   - Adequate labeled data is crucial for training accurate models. In some cases, obtaining a large and diverse dataset can be challenging.


2. **Imbalanced Datasets:**
   - When one class in a classification problem has significantly fewer instances than the others, the model may be biased towards the majority class.

3. **Overfitting:**
   - Overfitting occurs when a model learns the training data too well, capturing noise and outliers instead of general patterns. This can result in poor performance on new, unseen data.

4. **Underfitting:**
   - Conversely, underfitting occurs when a model is too simple to capture the underlying patterns in the data, leading to poor performance even on the training set.

5. **Feature Engineering:**
   - Selecting and engineering relevant features is crucial for model performance. Inadequate feature selection can lead to suboptimal results.

6. **Computational Complexity:**
   - Some algorithms, especially complex models like deep neural networks, may require significant computational resources and time for training.

7. **Bias and Fairness:**
   - Models trained on biased data may perpetuate and even exacerbate existing biases. Ensuring fairness and mitigating bias is an ongoing challenge.

8. **Interpretability:**
   - Complex models, such as deep neural networks, can be challenging to interpret. Understanding how and why a model makes a particular prediction is crucial in certain applications.

9. **Scalability:**
   - Ensuring that models can scale to handle larger datasets and increased computational demands is a concern, particularly in real-time applications.

10. **Data Quality:**
    - The quality of labeled data is paramount. Noisy or inaccurate labels can lead to the training of inaccurate models.

11. **Continuous Learning:**
    - Adapting models to changing data distributions over time (concept drift) is a challenge, especially in dynamic environments.

Addressing these challenges requires a combination of domain expertise, careful experimental design, and ongoing research in algorithm development and model evaluation techniques.

# Urdu my Smjy:
Supervised learning ke saath kuch challenges hote hain. Yeh kuch aam challenges hain:

1. **Data Ki Kami:**
   - Sahi labeled data ka hona bohot zaroori hai. Kabhi-kabhi large aur diverse dataset milana mushkil ho sakta hai.

2. **Imbalanced Datasets:**
   - Kabhi-kabhi ek class dusre se bahut kam instances contain karta hai, jisse model majority class ke taraf jhuk sakta hai.

3. **Overfitting:**
   - Overfitting tab hota hai jab model training data ko bahut zyada acche se sikhta hai, lekin yeh noise aur outliers ko bhi include kar leta hai. Isse naye data par performance kharab ho sakti hai.

4. **Underfitting:**
   - Overfitting ke ulte, underfitting tab hota hai jab model bahut simple hota hai aur data ke underlying patterns ko capture nahi kar pata.

5. **Feature Engineering:**
   - Relevant aur acche features ka chunav karna model ke liye bahut zaroori hai. Galat feature selection se model ki performance pe asar padta hai.

6. **Computational Complexity:**
   - Kuch algorithms, khaaskar deep neural networks jaise complex models, ke liye zyada computational resources aur time chahiye hota hai.

7. **Bias aur Fairness:**
   - Models jo biased data par train hote hain, woh biases ko barha sakte hain. Fairness aur bias ko control karna ek ongoing challenge hai.

8. **Interpretability:**
   - Kuch models, jaise deep neural networks, ko samajhna mushkil hota hai. Yeh samajhna ke model ek particular prediction kyun aur kaise karta hai, kuch applications ke liye zaroori hota hai.

9. **Scalability:**
   - Models ko large datasets aur increased computational demands ke sath handle karne ka guarantee karna bhi ek challenge hai.

10. **Data Quality:**
    - Labeled data ki quality bahut important hai. Noisy ya inaccurate labels se model galat sikh sakta hai.

11. **Continuous Learning:**
    - Models ko time ke saath changing data distributions ke liye adapt karna (concept drift) bhi ek challenge hai, khaaskar dynamic environments mein. 

In challenges ko handle karne ke liye domain knowledge, careful experimentation, aur algorithm development mein ongoing research ki zarurat hoti hai.


> **Conculssion**:
  In summary, supervised learning is a valuable tool with broad applications, but its success depends on the quality of labeled data, careful model design, and addressing potential challenges. Ongoing advancements in the field continue to enhance its capabilities and address its limitations.



  #      Chapter : 01(Linear Regression)


 ## ***1.Introduction of linear Regression:***

### ***1.1.1: Definition***
Linear regression is one of the simplest and Mostly commonly used types of Predictive Analysis in `Statistics` and `Machine Learning`

Linear regression is an algorithm that provides a linear relationship between an independent variable and a dependent variable to predict the outcome of future events. It is a statistical method used in data science and machine learning for predictive analysis

**(Urdu My Sikhy):**
Linear regression ek statistical tareeqa hai jo machine learning mein istemal hota hai taake ek ya zyada mustaqil variables (independent variables) aur ek dependent variable ke darmiyan taaluqat ka model banaya jaye. Is ka maqsad data ke behtareen fit hone wale ek linear equation ko daryaft karna hai, jisse ke dependent variable ke qiymat ko independent variables ke adadon ke zariye peish kiya ja sake

## ***1.1.2 Mathematical formula***

The mathematical formula for simple linear regression, where there is one independent variable, can be expressed as:

$[ Y = β_0 + β_1 \cdot X + ε ]$

Here:

- \( Y \) is the dependent variable (output),
- \( X \) is the independent variable (input),
- \( β_0 \) is the y-intercept (constant term),
- \( β_1 \) is the slope of the line (coefficient associated with the independent variable),
- \( ε \) represents the error term.

The goal of linear regression is to find the values of \( β_0 \) and \( β_1 \) that minimize the sum of squared differences between the observed (\( Y \)) and predicted (\( β_0 + β_1 \cdot X \)) values. This is typically achieved using the least squares method. The formula for \( β_1 \) is given by:

$[ β_1 = \frac{Cov(X, Y)}{Var(X)} ]$

And the formula for \( β_0 \) is derived from the equation of a line, \( Y = β_0 + β_1 \cdot X \), by rearranging it as:

\[ β_0 = \bar{Y} - β_1 \cdot \bar{X} \]

Here:

- \( Cov(X, Y) \) is the covariance between \( X \) and \( Y \),
- \( Var(X) \) is the variance of \( X \),
- \( \bar{Y} \) is the mean of \( Y \),
- \( \bar{X} \) is the mean of \( X \).

For multiple linear regression, where there are multiple independent variables, the formula extends to:

\[ Y = β_0 + β_1 \cdot X_1 + β_2 \cdot X_2 + \ldots + β_n \cdot X_n + ε \]

The coefficients \( β_0, β_1, β_2, \ldots, β_n \) are determined to minimize the sum of squared differences between the observed and predicted values.

## 1.1.3 Assumptions:
 Certainly! Here are the key assumptions of linear regression, explained briefly:

1. **Linearity:**
   - The relationship between variables is assumed to be linear.

2. **Independence of Errors:**
   - Errors (the differences between observed and predicted values) should be independent.

3. **Homoscedasticity:**
   - The variance of errors should be constant across all levels of the independent variable.

4. **Normality of Errors:**
   - Errors are assumed to be normally distributed.

5. **No Perfect Multicollinearity:**
   - Independent variables should not be perfectly correlated.

6. **No Autocorrelation:**
   - Residuals should not exhibit patterns over time.

7. **No Endogeneity:**
   - Independent variables are not influenced by errors.

8. **Additivity:**
   - Changes in one variable have an additive effect on the dependent variable.

Ensuring these assumptions helps maintain the reliability and validity of the linear regression model. Violations may lead to biased estimates and affect the model's accuracy.

## 1.1.3 Applications:
  Certainly! Here are six common and important applications of linear regression with brief explanations:

1. **Economics:**
   - *Application:* Predicting economic indicators like GDP based on factors such as investment and consumption.
  
2. **Finance:**
   - *Application:* Modeling stock prices and predicting returns, analyzing the relationship between financial variables.

3. **Healthcare:**
   - *Application:* Predicting patient outcomes and estimating the relationship between risk factors and health outcomes.

4. **Education:**
   - *Application:* Predicting student performance based on factors like study time, attendance, and socioeconomic status.

5. **Real Estate:**
   - *Application:* Predicting house prices based on features like size, location, and number of bedrooms.

6. **Marketing:**
   - *Application:* Estimating sales based on advertising expenditure, understanding the impact of marketing campaigns on customer behavior.


***Code is Here***
____
`# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# load the data tips from sns
df = sns.load_dataset('tips')
df.head()



# split the data into X and y
X = df[['total_bill']]
# scalar = MinMaxScaler()
# X = scalar.fit_transform(X)
y = df['tip']


# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# call the model
model = LinearRegression()

# train the model
model.fit(X_train, y_train)


# take out model intercept and slop, make an equation
print(model.intercept_)
print(model.coef_)
print('y = ', model.intercept_, '+', model.coef_, '* X')

# predict
y_pred = model.predict(X_test)


#evaluate the model
print('MSE: ', mean_squared_error(y_test, y_pred))
print('R2 = ', r2_score(y_test, y_pred))
print('RMSE = ', np.sqrt(mean_squared_error(y_test, y_pred)))


# plot the model and data
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.show()
`

