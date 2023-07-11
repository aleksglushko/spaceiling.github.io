### XGBoost regularization parameters study

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

Another good point to mention is that the model uses Weighted Quantile Sketch splitting technique. For every $k$-th feature values there is a corresponding second order gradient statistics $h_i$, that forms a multi-set 
```math
\mathcal{D}_{k} = \{(x_{1k}, h_1), (x_{2k}, h_2), ..., (x_{nk}, h_n)\},
```
here $x_{ik}$ represents the $i$-th element of $k$-th feature value. Then the following ranking function $r_k: \mathcal{R} \rightarrow [0, +\inf)$ for feature importance by the hessian values is used:

```math
r_k(z) = \frac{1}{\sum_{(x, h) \in \mathcal{D}_k} h} \sum_{(x, h) \in \mathcal{D_k}, x < z} h
```
which represents the proportion of instances whose feature value $k$ is smaller than $z$. The goal is to find candidate split points $\{s_{k1}, s_{k2}, ..., s_{kl} \}$ such that 
```math
|r_k(s_{k, j}) - r_k(s_{k, k+1})| < \epsilon, s_{k1} = \min_i x_{ik}, s_{kl} = \max_i x_{ik},
```
here $\epsilon$ is an approximation factor. The ranking function is used to prioritize the importance of different data points when selecting these candidates. 
It ranks data points based on the magnitude of their Hessian, because these values indicate how much each data point contributes to the curvature of the loss 
function. The split points $s_{k, j}$ are chosen such that the instances in each group have, on average, the same amount of influence over the final prediction 
of the model. This helps to ensure that the model pays approximately equal attention to the instances in each group when learning to predict the target variable.

Data points with a larger Hessian represent areas where the model's predictions are currently more uncertain, so it makes sense to prioritize considering these points as potential split candidates. In the loss function it is represented by the second order approximation term:

```math
\sum_{i=1}^n \frac{1}{2} h_i (f_i(x_i) - g_i/h_i)^2
```

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

**Min child weight**: defines the minimum sum of instance weights (also known as Hessian) needed in a child node. In simpler terms, if a proposed split results 
in a child node that has a sum of instance weights (Hessian) less than `min_child_weight`, then the split is discarded. In practice, it can be used to control 
the depth of the tree, as nodes that have a sum of instance weights less than `min_child_weight` are not split, and thus the tree depth can be controlled.

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

### Experiments and Results

The methodology for determining the optimal set of parameters revolves around leveraging the Hyperparameter Optimization (HPO) capabilities provided by AWS 
Sagemaker. In order to figure out the effect of regularization parameters, the results of using each regularization separately are presented.

| workload id                           | regularizaion     | max_depth | num_round | eta    | regularizaion | rmse (val) |
|---------------------------------------|-------------------|-----------|-----------|--------|---------------|------------|
| 292c737b-f648-45f5-8c40-ec5ef5e512cd  | no                | 11        | 289       | 0.007  |               | 3.767      |
|                                       | alpha             | 7         | 10        | 0.5    | 0.072         | 4.21       |
|                                       | lambda            | 11        | 328       | 0.007  | 0.001         | 3.768      |
|                                       | min_child_weight  | 32        | 127       | 0.018  | 35            | 3.786      |
|                                       | subsample         | 16        | 337       | 0.006  | 0.5           | 3.716      |
|                                       |                   |           |           |        |               |            |
| 732789ab-0769-4bf7-b9c3-97fd654c3963  | no                | 8         | 400       | 0.007  |               | 1.313      |
|                                       | alpha             | 8         | 447       | 0.011  | 47            | 1.309      |
|                                       | lambda            | 8         | 12        | 0.266  | 0.081         | 1.313      |
|                                       | min_child_weight  | 19        | 234       | 0.015  | 200           | 1.31       |
|                                       | subsample         | 9         | 152       | 0.021  | 0.5           | 1.311      |
|                                       |                   |           |           |        |               |            |
| 798fdddd-abfa-454b-be98-843d47c12291  | no                | 9         | 400       | 0.012  |               | 3.273      |
|                                       | alpha             | 37        | 297       | 0.014  | 120           | 3.257      |
|                                       | lambda            | 5         | 433       | 0.041  | 0.012         | 3.325      |
|                                       | min_child_weight  | 8         | 253       | 0.024  | 1             | 3.212      |
|                                       | subsample         | 5         | 394       | 0.066  | 0.683         | 3.303      |
|                                       |                   |           |           |        |               |            |
| 81799c6d-27c6-47f2-aaa5-9784b0ad5cf5  | no                | 5         | 159       | 0.009  |               | 4.299      |
|                                       | alpha             | 6         | 450       | 0.003  | 3.4           | 4.277      |
|                                       | lambda            | 5         | 396       | 0.003  | 0.0007        | 4.3        |
|                                       | min_child_weight  | 8         | 347       | 0.004  | 200           | 4.26       |
|                                       | subsample         | 6         | 430       | 0.003  | 0.69          | 4.24       |
|                                       |                   |           |           |        |               |            |
| bcefe84c-41f5-42cc-a811-f10d339542e9  | no                | 5         | 66        | 0.073  |               | 0.437      |
|                                       | alpha             | 5         | 244       | 0.033  | 0.0035        | 0.438      |
|                                       | lambda            | 5         | 428       | 0.014  | 0.0002        | 0.437      |
|                                       | min_child_weight  | 5         | 449       | 0.015  | 1             | 0.438      |
|                                       | subsample         | 5         | 157       | 0.032  | 0.88          | 0.438      |
|                                       |                   |           |           |        |               |            |
| ead2211f-2345-476a-9af2-53178e68929c  | no                | 8         | 87        | 0.029  |               | 1.391      |
|                                       | alpha             | 18        | 114       | 0.077  | 120           | 1.388      |
|                                       | lambda            | 9         | 10        | 0.27   | 0.0009        | 1.39       |
|                                       | min_child_weight  | 49        | 324       | 0.008  | 123           | 1.383      |
|                                       | subsample         | 9         | 424       | 0.007  | 0.5           | 1.383      |

