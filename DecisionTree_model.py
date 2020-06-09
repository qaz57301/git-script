import numpy as np

class DecisionNode():
    """
    class that represents a decision node or leaf in the decision tree
    :parameters
    feature_i: int
       Feature index which we want to use as the threshold measure.
    threshold: float
      The value that we will compare feature value at feature_i against to
      determine the prediction.
    value: float
      The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
      Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
      Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self,feature_i=None,threshold=None,
                 value=Node,true_branch=None,false_branch=None):
        self.feature_i = feature_i   #Index for the feature that is tested
        self.threshold = threshold   #Threshold value for feature
        self.value = value  #Value if the node is a leaf in the tree.
        self.true_branch=true_branch
        self.false_branch=false_branch

#super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    """
    Super class of RegressionTree and classificationTree.
    :parameters:
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    loss: function
        Loss function that is userd for Gradient Boosting models to calculate impurity.
    """
    def __init__(self,min_samples_split = 2,min_impurity = 1e-7,
                 max_depth = float("inf"),loss=None):
        self.root=None   #Root node in dec. tree
        #Minumum n of samples to justify split
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        #Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        #切割树的方法，gini，方差等
        self._impurity_calculation = None
        #Function to determine prediction of y at least.
        #树节点取值的方法，分类树：选取出现最多次数的值，
        self._leaf_value_calculation=None
        #If y is one-hot encoded(multi-dim) or not (one-dim)
        self.one_dim = None
        #If Gradient Boost
        self.loss = loss

    def fit(self,x,y,loss=None):
        """Build decision tree"""
        self.one_dim=len(np.shape(y))==1
        self.root=self._build_tree(x,y)
        self.loss = None

    def _build_tree(self,x,y,current_depth=0):
        """Recursive method which builds out the
        the decision tree and splits x and respective y
        on the feature of x which (based on impurity) best separates the data
        """
        largest_impurity = 0
        best_criteria = None   #Feature index and threshold
        best_sets = None  #subsets of the data

        #check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y,axis=1)

        #Add y as last column of x
        xy = np.concatenate((x,y),axis=1)

        n_samples,n_features = np.shape(x)
        if n_samples>=self.min_samples_split and current_depth<=self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All value of feature_i
                feature_values = np.expand_dims(x[:,feature_i],axis=1)
                unique_values = np.unique(feature_values)

                #Iterate through all unique values of feature column i and
                #calculate the impurity
                for threshold in unique_values:
                    # Divide x and y depending on if the feature value of x at index feature_i
                    xy1,xy2 = divide_on_feature(xy,feature_i,threshold)
                    if len(xy1)>0 and len(xy2)>0:
                        #select the y-values of the two sets
                        y1 = xy1[:,n_features:]
                        y2 = xy2[:,n_features:]

                        #calculate impurity
                        impurity = self._impurity_calculation(y,y1,y2)

                        #If this threshold resulted in a higher information gain than previously
                        """
                        recorded save the threshold value and the feature index
                        """
                        if impurity>largest_impurity:
                            largest_impurity=impurity
                            best_criteria = {"feature_i":feature_i,"threshold":threshold}
                            best_sets={
                                "leftx":xy1[:,:n_features],
                                "lefty":xy1[:,n_features:],
                                "rightx":xy2[:,:n_features],
                                "righty":xy2[:,n_features:]
                            }
        if largest_impurity>self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftx"],best_sets["lefty"],current_depth+1)
            false_branch = self._build_tree(best_sets["rightx"],best_sets["righx"],current_depth+1)
            return DecisionNode(feature_i=best_criteria["feature_i"],threshold=
                                best_criteria["threshold"],true_branch=true_branch,false_branch=false_branch)


































