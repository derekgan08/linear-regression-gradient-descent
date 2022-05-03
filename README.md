# CPC251 Assignment 1: Linear Regression using Gradient Descent in QSAR Biodegradation

## Problem Statements
Linear regression is a fundamental machine learning algorithm used for predictive modeling. One of the main challenges in building an accurate regression model is properly estimating the model's weights or parameters. This project focuses on solving this challenge using the gradient descent optimization technique, which iteratively adjusts the weights to minimize the model's loss function.

By implementing gradient descent, the project aims to demonstrate how to efficiently train a linear regression model using Python. This software can be helpful for data scientists and machine learning practitioners to understand the core concepts of linear regression and optimization.

## Project Overview
This project implements a linear regression model that utilizes the gradient descent algorithm to iteratively update the model’s weights. The model is trained on a dataset split into an 80:20 ratio for training and testing. The gradient descent algorithm works by minimizing the loss function, which helps in finding the optimal model parameters.

Key functionalities of the project:
- **Training**: The `train_model` function performs gradient descent to estimate the weights (parameters) of the linear regression model.
- **Prediction**: The `prediction` function uses the trained weights to make predictions on unseen data.
- **Loss Function**: The `loss_fn` function computes the loss using the mean squared error metric to evaluate model performance.
- **Visualization**: The project includes a plot of the loss history over training epochs, allowing users to observe the convergence of the gradient descent algorithm.
  
The project is designed to demonstrate a fundamental machine learning technique and is useful for anyone wanting to understand how gradient descent works in the context of linear regression.

## Key Features
- Implements gradient descent for training a linear regression model.
- Provides a clear, step-by-step implementation of the algorithm.
- Visualizes the loss history during training to track the optimization process.
- Includes evaluation of the model using key metrics such as mean squared error.
- Splits the dataset into training and test sets for proper model evaluation.

## Technologies Used
- **Python 3.x**: Primary programming language
- **NumPy**: For numerical operations and matrix manipulations
- **Matplotlib**: For plotting training loss and scatter plots for predictions
- **Pandas**: For data manipulation and loading the dataset
- **Scikit-learn**: For splitting the dataset into training and test sets