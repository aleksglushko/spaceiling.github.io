### XGBoost regularizations ablation study

This report aims to provide a an examination of some regularizations in XGBoost, with a focus on empirical analysis. We will first introduce the concept of 
regularization in the context of machine learning and explore how different regularization techniques are implemented in XGBoost. Following a detailed description of our 
experimental methodology, we will present the results of a series of experiments conducted to evaluate the impact of different regularization parameters on the performance of 
an XGBoost model.

This report is structured as follows: Section "Regularizations in XGBoost" provides 
a theoretical background of regularization and its role in XGBoost. In Section "Methodology", we outline
the datasets used and our approach to parameter tuning and model evaluation. The experimental results are presented and discussed in "Results and Experiments" section.

### Introduction

In the context of XGBoost, the loss function is differentiable and typically selected based on the problem at hand (e.g., logistic loss for classification, squared error for regression).
Commonly, it can be represented as:

```math
\mathcal{L} = \sum l(y_i, \hat{y}_i) + \gamma T + \frac{1}{2}\lambda ||\omega||^2
```
here $l$ is some loss function, $\gamma$ controls complexity of the tree structure, $\lambda$ controls scale of the leaf scores.

Once we define the objective function, the XGBoost algorithm aims to find the model that minimizes this objective. Given that XGBoost builds an ensemble of decision trees, 
the model needs an effective way to construct these trees. That's where the split finding function comes into play.

The split finding function is a greedy algorithm that decides the optimal split at each node in a decision tree. It calculates a gain for each potential split and chooses the one that 
yields the highest gain. The gain is a measure of the reduction in the objective function achieved by the split. Imagine $I_L$ and $I_R$ are the instance sets of left and right nodes after the plit. Letting $I = I_L \cup I_R$, then the loss reduction after the split can be represented as:

```math
\mathcal{L}_{\text{split}} = \frac{1}{2} \big[ \frac{(\sum_{i \in I_L g_i})^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R g_i})^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I g_i})^2}{\sum_{i \in I} h_i + \lambda} \big],
```
here $g_i$ and $h_i$ are the first and second order gradient statistics on the loss function. The derivation of this function can be looked up in the [paper](https://arxiv.org/pdf/1603.02754.pdf).

In the following section, we will delve into regularization in XGBoost, exploring various forms of regularization techniques and the role they play in controlling model complexity and 
combating overfitting.

### Regularizations in XGBoost

**L1 Regularization** (`alpha`): also known as Lasso Regression, it adds an L1 penalty to the loss function and is equal to the absolute value of the magnitude of the coefficients. By 
the coefficients we consider leaf weights of a trained tree. This type of regularization encourages sparser models by pushing some of the leaf weights to be exactly zero. This sparsity 
can make the model more interpretable and can help prevent overfitting by reducing the model's complexity. The larger the value of `alpha`, the more aggressive this penalty will be.
It is important to note that it doesn't directly encourage trees with fewer leaves or less depth. Rather, it affects the values of the weights associated with each leaf.

The cost function to be minimized becomes:

```math
loss = \sum_{i=0}^n l(y_i, X_i \beta) + \alpha \sum_{j=0}^m |\beta_j|,
```
where $l$ is a loss function, $y_i$ is a ground truth, $X$ is a features input, $\beta$ - coefficients. Considering the square error loss, the equation becomes as follows:

```math
loss = \sum_{i=0}^n (y_i - X_i \beta)^2 + \alpha \sum_{j=0}^m |\beta_j|,
```

**L2 Regularization** (`lambda`): also known as Ridge Regression, L2 regularization adds a penalty equal to the square of the magnitude of the coefficients. This type of regularization tends 
to spread the coefficient values out more equally. L1 regularization can be more robust to small changes in input data, because it allows the model to ignore less important features by 
setting their coefficients to zero. However, if you have two highly correlated features, L1 regularization tends to select one arbitrarily. L2 regularization is more stable in that it 
tends to assign similar coefficients to correlated features. Furthemore, the L2 norm penalizes large coefficients more heavily than the L1 norm does. Also it tends not to produce sparsity 
because the penalty decreases as a coefficient approaches zero, making it less beneficial to set a coefficient to zero than it is with L1 regularization.

```math
loss = \sum_{i=0}^n (y_i - X_i \beta)^2 + \alpha \sum_{j=0}^m \beta^2_j,
```

**Minimum loss reduction** (`gamma`): controls the complexity of inividual trees in the ensemble. It provides a threshold for the reduction in the loss required
to make an additional split on a leaf node of the tree. When considering adding a new split to a leaf node in the tree, XGBoost calculates the reduction in loss
that would result from the split. If this reduction in loss is less than `gamma`, then the algorithm decides not to make the split. In other words, the split is 
made only if it decreases the loss by at least a value of `gamma`. Therefore, larger values of `gamma` will result in fewer splits and thus simpler, more 
conservative models. On the other hand, smaller values of `gamma` allow more complex models with more splits.

**Min child weight**: defines the minimum sum of instance weights (also known as Hessian) needed in a child node. In simpler terms, it corresponds to the minimum 
number of instances needed to be in each node. In practice, it can be used to control the depth of the tree, as nodes that have a sum of instance weights less than 
`min_child_weight` are not split, and thus the tree depth can be controlled. If `min_child_weight` is set to a large value, it could lead to underfitting, as 
the algorithm would be constrained to create only very broad splits. On the other hand, setting it too low could lead to the model capturing too much noise in the data and thus overfitting.

**Subsample**: controls the fraction of rows in the training data to be used for any given tree. This is a form of row subsampling, similar to the technique used in Random Forests.
The idea is to add some randomness to the model training process to make the model more robust and prevent overfitting. It does this by creating a sort of "ensemble" effect 
within each individual tree, where each tree only gets to "see" a random subset of the data.
The subsample parameter takes values between 0 and 1:

- A value of 1 means that we use all rows (i.e., no row subsampling).
- A value of 0.5 means that each tree uses 50% of the rows selected randomly.
- A value of 0 would mean that no rows are used, which of course wouldn't be useful.

**Colsample**: the colsample_bytree and colsample_bylevel parameters control the subsampling of columns (features) used for constructing each tree or split in a level.

- A value of 1 means that we use all columns (i.e., no column subsampling).
- A value of 0.5 means that each tree or level uses 50% of the columns selected randomly.
- A value of 0 would mean that no columns are used, which of course wouldn't be useful.

### Methodology
The methodology for determining the optimal set of parameters revolves around leveraging the Hyperparameter Optimization (HPO) capabilities provided by AWS Sagemaker. In order to 
figure out the affect of regularization parameters below the results of leveraging each regularization separately are presented.

### Experiments and Results

