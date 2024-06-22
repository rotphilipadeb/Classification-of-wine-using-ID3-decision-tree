# Classification-of-wine-using-ID3-decision-tree
Decision Trees (DTs) are a non-parametric supervised learning method used for  classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. 


<img src="different_classes_of_wine.webp" alt="Decision Tree Model" width="1000" height="620">
<h2>Background</h2>
<p>Decision Trees (DTs) are a non-parametric supervised learning method utilized for both classification and regression tasks. The objective of this project is to build a model that predicts the value of a target variable by learning straightforward decision rules from the features of the dataset. After training, a decision tree is obtained. In practice, a predictive decision tree model incrementally selects the optimal decisions to split on, evaluated based on the entropy principle, to produce an output classification from the input data. For this assessment, you will describe a new problem and use certain machine learning Python modules to create an ID3 predictive decision tree model. Additionally, you will provide a visualization to better understand the classification process..</p>

<p>This Task will measure your ability to</p>
<ol>
  <li>study a problem that can be tackled by an artificial intelligence method, e.g., a decision tree;</li>
  <li>implement the decision tree using the given instructions;</li>
   <li>visualize and analyse the decision tree.</li>
</ol>
 
<p>
The objective of this assessment is to utilize the pandas and scikit-learn libraries to implement the ID3 decision tree machine learning algorithm to create a classifier for a specific problem. Using the visualization of the trained tree, you should gain a reinforced understanding of the core principles of decision trees and how their split evaluations operate.</p>
<h2>Problem Description</h2>
<p>You will be creating a decision tree that will predict wine classes based on provided attributes.
Imagine that you are a wine producer compiling data for a study. The data is the results of a 
chemical analysis of wines grown in the same region in Italy by three different cultivators. There are thirteen different measurements taken for different constituents found in the three types of wine, class_0, class_1, and class_2.</p>

<p>Those thirteen different measurements include: Alcohol, Malic acid, Ash, Alcalinity of ash, 
Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color 
intensity, Hue, OD280/OD315 of diluted wines and Proline.</p>
<p>Therefore, the model’s input parameters can be Those thirteen different measurements and the model output should be the wine classes.</p>
<h2>Implementation instructions</h2>
<h3>Assignment dependency installation</h3>
<p>You will have to install the following required dependency:</p>
<ol>
  <li>scikit-learn</li>
  <li>pandas</li>
  <li>matplotlib</li>
</ol>

<h2>Imports</h2>
<p>You will be utilizing several well-known machine learning Python modules in this task. These steps facilitate the implementation of our decision tree and include numerous learning tools to enhance your understanding.</p>
<p><i>import pandas as pd</i></p>
<p><i>import sklearn </i></p>
<p><i>import matplotlib.pyplot as plt</i></p>
<p>Load the dataset and Format the training/testing data
You will be using the following codes to load the dataset:</p>

# Load wine dataset
from sklearn.datasets import load_wine
data = load_wine()
<p>The pandas Python module is a highly powerful and frequently used data analysis tool in various forms of machine learning. It allows you to easily store and manipulate large datasets and is highly compatible with other machine learning tools and modules. In this task, you will store your dataset in a pandas DataFrame. A DataFrame is similar to a dictionary in standard Python but offers many additional useful features. To add data to the DataFrame, specify the data keys and their corresponding values..</p>
<p>Next, we need to create a training set to train the classifier and a test set to evaluate the classifier's performance. This can be done easily by splitting the collected data into two groups. For instance, if we have 100 records, we can use 80 records as the training set and the remaining 20 records as the test set.</p>
<h2>Train the decision tree</h2>
<p>Once your data is correctly formatted, you can proceed to create the decision tree. Begin by creating a new DecisionTreeClassifier from the scikit-learn library, using 'entropy' as the criterion for information gain. Scikit-learn is a powerful machine learning framework, and you will use its DecisionTreeClassifier class to create and train your decision tree. Follow the instructions provided in this link: scikit-learn decision trees. To train the decision tree, call the fit method on the classifier object..</p>
<h2>Create a graph visualisation</h2>
<p>Next, use the plt.figure method to create a graphical representation (dot data) of the trained decision tree. Save this visualization as a PDF named output_graph in your working directory.</p>
<p>In your report, provide a detailed discussion of the generated graph visualization. Ensure to include the output_graph in your Word file. Here are some useful details you may need for visualizing a decision tree:</p>
<p>https://scikit-learn.org/stable/modules/tree.html and https://scikitlearn.org/stable/modules/generated/sklearn.tree.plot_tree.html</p>
<h2>Test the decision tree</h2>
<p>To test your developed model, you will have to pass the test set into your trained 
decision tree classifier. Input this test set into the trained model. Calculating the 
classification accuracy is needed.</p>
<h2>Detailed program specifications</h2>
<p>According to the above descriptions, this task requires you to fulfill the following 
objectives:</p>
<ol>
  <li>Load the wine dataset correctly, and split it into train/test sets appropriately;</li>
  <li>Appropriate implementation of a decision tree for solving the classification task 
using the loaded dataset;</li>
  <li>Train decision tree correctly, obtaining good test results (the actual test 
performance should be revealed in report);</li>
  <li>Visualize the trained decision tree correctly. </li>
</ol>

