### XGBoost regularizations ablation study

This report aims to provide a comprehensive examination of the role of regularization in XGBoost, with a focus on empirical analysis. We will first introduce the concept of 
regularization in the context of machine learning and explore how different regularization techniques are implemented in XGBoost. Following a detailed description of our 
experimental methodology, we will present the results of a series of experiments conducted to evaluate the impact of different regularization parameters on the performance of 
an XGBoost model.

This report is structured as follows: Section 1 provides a theoretical background of regularization and its role in XGBoost. In Section 2, we outline our methodology, describing 
the datasets used and our approach to parameter tuning and model evaluation. The experimental results are presented and discussed in Section 3. Finally, we summarize our findings 
and draw conclusions in Section 4.

### Regularizations in XGBoost

**L1 Regularization** (`alpha`): Also known as Lasso Regression, it adds an L1 penalty to the loss function and is equal to the absolute value of the magnitude of the coefficients. By 
the coefficients we consider leaf weights of a trained tree. This type of regularization encourages sparser models by pushing some of the leaf weights to be exactly zero. This sparsity 
can make the model more interpretable and can help prevent overfitting by reducing the model's complexity. The larger the value of `alpha`, the more aggressive this penalty will be.
It is important to note that it doesn't directly encourage trees with fewer leaves or less depth. Rather, it affects the values of the weights associated with each leaf.

The cost function to be minimized becomes:

```math
loss = \sum_{i=0}^n l(y_i, X_i \beta) + \alpha \sum_{j=0}^m |\beta_j|,
```
where $l$ is a loss function, $y_i$ is a ground truth, $X$ is a features input, $\beta$ - coefficients.

**L2 Regularization** (`lambda`): Also known as Ridge Regression, L2 regularization adds a penalty equal to the square of the magnitude of the coefficients. This type of regularization tends 
to spread the coefficient values out more equally. L1 regularization can be more robust to small changes in input data, because it allows the model to ignore less important features by 
setting their coefficients to zero. However, if you have two highly correlated features, L1 regularization tends to select one arbitrarily. L2 regularization is more stable in that it 
tends to assign similar coefficients to correlated features. Furthemore, the L2 norm penalizes large coefficients more heavily than the L1 norm does. Also it tends not to produce sparsity 
because the penalty decreases as a coefficient approaches zero, making it less beneficial to set a coefficient to zero than it is with L1 regularization.

```math
loss = \sum_{i=0}^n l(y_i, X_i \beta) + \alpha \sum_{j=0}^m \beta^2_j,
```

...
more regularization?

### Methodology
TODO: Expand
we are using Optuna + Ray Tune + ranges established earlier

### Experiments and Results

