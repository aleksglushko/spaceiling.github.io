### Mastering XGBoost: Introduction, Parameters, and Tuning with regards SageMaker Pipeline framework
   _Abstract: In this post, the gradient boosting and hyperparameters of the model are the main discussion points, since these are the ones used in the SageMaker Pipelines implementation. It is not intended to be the full guide about the XGBoost models. First of all, the main benefits of the model are mentioned and an objective function is considered. Then, the splitting methods and hyperparameters are discussed. Later the XGBoost framework in Sagemaker is discussed._
   
XGBoost, which stands for eXtreme Gradient Boosting, is a machine learning algorithm, known for its speed and high-quality performance. It 
represents an evolved variant of ensemble learning techniques. In such methods, every new model in the sequence is trained with a unique target: the residuals or 
differences between the predictions of the existing model and the actual ground truth. This practice allows the new model to learn from the mistakes of its 
predecessor. Ultimately, the final prediction is produced by aggregating the predictions from all the models in the ensemble, that are basically the scores in the corresponding leaves.

The distinguishing characteristic of XGBoost lies in its incorporation of a regularization term in the objective function, along with the application of a 
sophisticated second-order Taylor expansion for approximation. Here are some advantages that XGBoost holds over other models, which we will explore in more detail 
later in the post:
- Regularization: XGBoost introduces an added regularization term in the objective function. This term, which factors in both the number of leaves in a tree and the 
scores assigned to those leaves, equips the model with a means to manage its complexity and counter overfitting. As a result, the model's approach is more 
conservative.
- Sparsity Awareness: XGBoost comes with an innate ability to handle sparse features in inputs. It attempts to assign a direction for these missing values during the 
tree building phase, instead of constant imputation.
- Weighted Quantile Sketch: This algorithm is employed by XGBoost to find the optimal split points in weighted datasets effectively.
- Tree Pruning: Unlike Gradient boosting, which halts the splitting of a node once it encounters a negative loss, XGBoost takes a different route. It grows the tree 
to its maximum depth and subsequently prunes the tree in a backward manner. Any splits that do not contribute to a positive gain are removed.

##### Regularized learning objective
For a given data set $X$, a tree ensemble model uses $K$ additive functions to predict the output.
```math
\hat{y}_i = \sum_{k=1}^K f_k(x_i), f_k \in F,
```
where $F$ is the space of regression trees. Unlike decision trees, each regression tree contains a continious score on each of the leaf. To learn the set of functions
used in the model, the following _regularized_ objective function should be minimized.
```math
L = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k),
```
where $`\Omega(f) = \gamma T + \frac{1}{2}\lambda ||\omega||^2,`$ here $l$ is a differentiable convex loss function that measures the difference between the 
prediction 
$\hat{y}_i$ and the target $y_i$. T is the number of leaves in the tree. Each $f_k$ corresponds to an independent tree structure $q$ and leaf weights $w$.

##### Weighted Quantile Sketch
For split findings _Weighted Quantile Sketch_ method is used. Since, the _Exact greedy algorithm_ takes too much computational memory, an approximation method takes 
place in the tree-based methods. The traditional approximation algorithm would solve this by picking a set of candidate split points up based on the percentiles of 
feature values, but this doesn't account for instance weights. In contrast, the _Weighted Quantile Sketch_ algorithm takes into account the weights of instances when 
choosing candidate split points. Neverthereless, the weights are not specified by XGBoost model. It maintains a sketch (a data structure) of the data that allows it 
to approximate the weighted quantiles. By doing this, it's better able to handle datasets where some instances are considered more important than others. 

##### Sparsity awaraness
When a tree is being constructed, and a split is being considered at a node for a feature that has missing values, XGBoost essentially constructs two paths: one path 
for samples where that feature's value is missing and one path for samples where that feature's value is present. It then compares the output (in terms of the loss 
function being optimized) of assigning all samples with missing values to the left versus the right child node and chooses the direction that optimizes the objective 
most. The chosen direction for the missing values (either left or right) is then stored as default direction for missing values in the tree structure. When making 
predictions, if the algorithm encounters a missing value for this feature, it routes it in the stored default direction.

##### XGBoost in [SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
Current supported verions:
- Framework (open source) mode: 1.0-1, 1.2-1, 1.2-2, 1.3-1, 1.5-1, 1.7-1
- Algorithm mode: 1.0-1, 1.2-1, 1.2-2, 1.3-1, 1.5-1, 1.7-1

[Hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html):
- seed: Random number seed. Optional Valid values: integer Default value: 0
- objective: Specifies the learning task and the corresponding learning objective. Examples: reg:logistic, multi:softmax, reg:squarederror. Optional Valid values: 
String Default value: "reg:squarederror"
- alpha: L1 regularization term on weights. Increasing this value makes models more conservative. Optional Valid values: Float. Default value: 0
- lambda: L2 regularization term on weights. Increasing this value makes models more conservative. Optional Valid values: Float. Default value: 1
- eta: Step size shrinkage used in updates to prevent overfitting. After each boosting step, you can directly get the weights of new features. The eta parameter 
actually shrinks the feature weights to make the boosting process more conservative. Optional Valid values: Float. Range: [0,1]. Default value: 0.3
- rate_drop: The dropout rate that specifies the fraction of previous trees to drop during the dropout. Optional Valid values: Float. Range: [0.0, 1.0]. Default 
value: 0.0
- gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm is. Optional Valid 
values: Float. Range: (0, inf) Default value: 0. 
   - When gamma is 0 or close to 0 (the default value in XGBoost), then almost any split that improves the model's accuracy on the training data is accepted, 
   potentially leading to a large and complex tree that may overfit the training data.
   - As gamma increases, the algorithm becomes more conservative. The tree becomes more constrained as it has to consider more seriously whether the gain from each 
   additional split is worth it. Hence, splits that result in only a small improvement in the loss will be discouraged, leading to simpler models and fewer splits.
- subsample: Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees. This prevents 
overfitting. Optional Valid values: Float. Range: [0,1]. Default value: 1
- eval_metric: rmse: for regression error: for classification map: for ranking
- booster: Which booster to use: gbtree, gblinear or dart. Default value: "gbtree"
   - gbtree: This option uses tree-based models as the base learners. Each iteration adds a new tree that corrects the errors made by the ensemble of the previous 
   trees. This is the default option, and it is the one that typically provides the best performance for a wide range of datasets.
   - gblinear: This option uses linear models as the base learners. Each iteration adds a new linear model that corrects the residual errors made by the ensemble of 
   the previous linear models. This option can be useful for high-dimensional sparse data, but it typically underperforms compared to "gbtree" for most datasets.
   - dart: This option also uses tree-based models as the base learners, similar to "gbtree". However, it applies a modification to the boosting process called 
   "Dropout Additive Regression Trees", which can make the model more robust to noise in the data. It essentially involves "dropping out", or temporarily removing, 
   some of the trees during the training process to prevent overfitting. 
- tree_method: The tree construction algorithm used in XGBoost. Optional Valid values: One of auto, exact, approx, hist, or gpu_hist. Default value: auto. There are 
several options for the tree_method parameter:
   - auto: The algorithm will choose the best method based on the characteristics of the data. This is the default value.
   - exact: This method builds the tree exactly as specified by the algorithm, and can be slower and use more memory than the other methods.
   - approx: This method builds an approximate tree. It can be faster and use less memory than exact, but the resulting model may be slightly less accurate.
   - hist: This method builds a tree using a fast histogram-based method, which can be faster and use less memory than exact. It's generally a good choice for large 
   datasets.
   - gpu_exact: This is the exact method implemented on the GPU. It can be faster than exact if you have a compatible GPU.
   - gpu_hist: This is the histogram-based method implemented on the GPU. Like gpu_exact, it can be faster than its CPU counterpart if you have a compatible GPU.
- normalize_type: Type of normalization algorithm. Optional Valid values: Either tree or forest. Default value: tree
   - tree: This normalization type divides the prediction result by the number of dropped trees. It's as if each tree that wasn't dropped gets an equal vote in the 
   final prediction.
   - forest: This normalization type doesn't consider the number of dropped trees, and instead divides the prediction result by all trees. This means that even the 
   dropped trees are considered in the normalization, which can make the model more conservative in its predictions.
- grow_policy: Controls the way that new nodes are added to the tree. Currently supported only if tree_method is set to hist. Optional Valid values: String. Either 
"depthwise" or "lossguide". Default value: "depthwise"
   - depthwise: This is the default option. In depthwise growth, the algorithm will split at all nodes of a given depth before progressing to the next depth level. 
   This results in balanced trees where each level of the tree is fully populated with splits before the next level is started. Depthwise growth policy can lead to 
   simpler and more interpretable trees.
   - lossguide: In lossguide growth, the algorithm will choose the node with the highest loss change to split first. This can result in unbalanced trees, where some 
   branches of the tree might go deeper before others start. Lossguide growth policy is more flexible and can potentially create more complex models that may fit the 
   training data better.
- max_depth: Maximum depth of a tree. Increasing this value makes the model more complex and likely to be overfit. 0 indicates no limit. A limit is required when 
grow_policy=depth-wise.
- max_leaves: Maximum number of nodes to be added. Relevant only if grow_policy is set to lossguide. Optional Valid values: Integer. Default value: 0
- colsample_bylevel: Subsample ratio of columns for each split, in each level. Optional Valid values: Float. Range: [0,1]. Default value: 1 (all features are used)
   In simpler terms, when building each new level of the tree, _colsample_bylevel_ determines the fraction of the total number of features that will be randomly 
   chosen to make the decision about where to split the data.
   For example, if you have 10 features and you set colsample_bylevel to 0.6, each new level of each tree will be built using only 6 randomly selected features out of 
   the 10.
   This has two main benefits:
      - It helps prevent overfitting by adding randomness into the model building process, which can increase the diversity of the tree structures in your ensemble of 
      trees.
      - It can make the model training process faster, especially if you have a large number of features, as each split decision now needs to consider fewer features.
- colsample_bynode: Subsample ratio of columns from each node. Optional Valid values: Float. Range: (0,1]. Default value: 1
- colsample_bytree: Subsample ratio of columns when constructing each tree. Optional Valid values: Float. Range: [0,1].
- early_stopping_rounds: The model trains until the validation score stops improving. Validation error needs to decrease at least every early_stopping_rounds to 
continue training. SageMaker hosting uses the best model for inference. Optional Valid values: Integer. Default value: -
- csv_weights: When this flag is enabled, XGBoost differentiates the importance of instances for csv input by taking the second column (the column after labels) in 
training data as the instance weights, to provide some features more importance. Optional Valid values: 0 or 1 Default value: 0
- max_delta_step: Maximum delta step allowed for each tree's weight estimation. When a positive integer is used, it helps make the update more conservative. The 
preferred option is to use it in logistic regression. Set it to 1-10 to help control the update. Default is 0. This parameter is particularly useful in logistic 
regression, where it can help control the updates when the classes are extremely imbalanced. 
- min_child_weight: Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less 
than min_child_weight, the building process gives up further partitioning. In linear regression models, this simply corresponds to a minimum number of instances 
needed in each node. The larger the algorithm, the more conservative it is.
- process_type: The type of boosting process to run. Optional Valid values: String. Either "default" or "update". Default value: "default". default: 
   - This is the standard process, which means each round of training will build a new tree.
   - update: This means the process will start from an existing model and attempt to update its structure with new trees. This can be useful if you have a pre-
   existing trained XGBoost model and you'd like to continue training it on new data.
- sample_type: Type of sampling algorithm for DART booster. Either uniform or weighted. Default value: uniform
   - uniform: This means that each tree in the model has an equal probability of being dropped out during training. In other words, the dropout process doesn't take 
   into account the performance of the individual trees. It's the equivalent of uniform dropout in neural networks, where each neuron has the same probability of 
   being dropped.
   - weighted: This means that the probability of a tree being dropped is proportional to its contribution to the model. Trees that contribute more to the model 
   (those that make larger improvements to the prediction error) have a higher probability of being dropped out.
- interaction_constraints: Specify groups of variables that are allowed to interact. Optional Valid values: Nested list of integers. Each integer represents a 
feature, and each nested list contains features that are allowed to interact e.g., [[1,2], [3,4,5]]. Default value: None
- max_bin: Maximum number of discrete bins to bucket continuous features. Used only if tree_method is set to hist. Optional Valid values: Integer. Default value: 256
- base_score: The initial prediction score of all instances, global bias. Optional Valid values: Float. Default value: 0.5. In case the training set is not balanced.
- lambda_bias: L2 regularization term on bias. Optional Valid values: Float. Range: [0.0, 1.0]. Default value: 0
- num_class: The number of classes. Required if objective is set to ´multi:softmax´ or ´multi:softprob´. Valid values: Integer.
- deterministic_histogram: When this flag is enabled, XGBoost builds histogram on GPU deterministically. Used only if tree_method is set to gpu_hist. For a full list 
of valid inputs, please refer to XGBoost Parameters. Optional Valid values: String. Range: "true" or "false". Default value: "true"
- monotone_constraints: Specifies monotonicity constraints on any feature. Optional Valid values: Tuple of Integers. Valid integers: -1 (decreasing constraint), 0 (no 
constraint), 1 (increasing constraint). E.g., (0, 1): No constraint on first predictor, and an increasing constraint on the second. (-1, 1): Decreasing constraint on 
first predictor, and an increasing constraint on the second. Default value: (0, 0)
- nthread: Number of parallel threads used to run xgboost. Optional Valid values: Integer. Default value: Maximum number of threads.
- one_drop: When this flag is enabled, at least one tree is always dropped during the dropout. Optional Valid values: 0 or 1 Default value: 0
- refresh_leaf: This is a parameter of the 'refresh' updater plug-in. When set to true (1), tree leaves and tree node stats are updated. When set to false(0), only 
tree node stats are updated. Optional Valid values: 0/1 Default value: 1
- scale_pos_weight: Controls the balance of positive and negative weights. It's useful for unbalanced classes. A typical value to consider: sum(negative cases) / 
sum(positive cases). Optional Valid values: float Default value: 1
- skip_drop: Probability of skipping the dropout procedure during a boosting iteration. Optional Valid values: Float. Range: [0.0, 1.0]. Default value: 0.0
- tweedie_variance_power: Parameter that controls the variance of the Tweedie distribution. Optional Valid values: Float. Range: (1, 2). Default value: 1.5
- updater: A comma-separated string that defines the sequence of tree updaters to run. This provides a modular way to construct and to modify the trees. For a full 
list of valid inputs, please refer to XGBoost Parameters. Optional Valid values: comma-separated string. Default value: grow_colmaker, prune
- verbosity: Verbosity of printing messages. Valid values: 0 (silent), 1 (warning), 2 (info), 3 (debug). Optional Default value: 1


