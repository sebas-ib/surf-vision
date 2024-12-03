import numpy as np
from collections import Counter

class DecisionTree:
    '''
    A simple decision tree classifier based on sklearn's DecisionTreeClassifier
    '''

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        '''
        Calculate the Gini Impurity for a list of classes
        '''
        _, counts = np.unique(y, return_counts=True)

        return 1 - np.sum((counts / len(y)) ** 2)

    def _split(self, X, y, feature, threshold):
        '''
        Split the dataset based on a feature and a threshold
        '''
        left_mask = X[:, feature] <= threshold

        return (X[left_mask], y[left_mask], X[~left_mask], y[~left_mask])

    def _best_split(self, X, y):
        '''
        Find the best feature and threshold to split on
        '''
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                _, y_left, _, y_right = self._split(X, y, feature, threshold)
                gini = (len(y_left) * self._gini(y_left) + len(y_right) * self._gini(y_right)) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        '''
        Recursively build the decision tree
        '''
        if len(np.unique(y)) == 1 or (self.max_depth and depth == self.max_depth):
            return Counter(y).most_common(1)[0][0]

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1)
        }

    def fit(self, X, y):
        '''
        Build the decision tree using the training data
        '''
        self.tree = self._build_tree(X, y)

    def _predict_single(self, x, tree):
        '''
        Predict the class for a single sample
        '''
        if not isinstance(tree, dict):
            return tree

        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])

        return self._predict_single(x, tree['right'])

    def predict(self, X):
        '''
        Predict the classes for a dataset
        '''
        return np.array([self._predict_single(x, self.tree) for x in X])

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def _bootstrap_sample(self, X, y):
        '''
        Create a bootstrap sample of the data
        '''
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)

        return X[idxs], y[idxs]

    def fit(self, X, y):    
        '''
        Train the random forest model by fitting each tree to a bootstrap sample of the data
        '''
        X = np.array(X)
        y = np.array(y)
        
        np.random.seed(self.random_state)
        self.trees = []
        
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        self.feature_importances_ = np.zeros(X.shape[1])
        
        for tree in self.trees:
            self._update_feature_importances(tree.tree, 1.0)
        self.feature_importances_ /= self.n_estimators

    def _update_feature_importances(self, node, weight):
        '''
        Update feature importances recursively
        '''
        if isinstance(node, dict):
            self.feature_importances_[node['feature']] += weight
            left_size = self._get_node_size(node['left'])
            right_size = self._get_node_size(node['right'])
            total_size = left_size + right_size
            if total_size > 0:
                left_weight = weight * left_size / total_size
                right_weight = weight * right_size / total_size
                self._update_feature_importances(node['left'], left_weight)
                self._update_feature_importances(node['right'], right_weight)

    def _get_node_size(self, node):
        '''
        Get the size of a node (number of samples)
        '''
        if isinstance(node, dict):
            return self._get_node_size(node['left']) + self._get_node_size(node['right'])
        else:
            return 1

    def predict(self, X):
        '''
        Predict the classes using majority voting
        '''
        X = np.array(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        return np.array([Counter(pred).most_common(1)[0][0] for pred in tree_preds.T])

    def predict_prob(self, X):
        '''
        Predict class probabilities
        '''
        X = np.array(X)
        
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        probs = []

        for pred in tree_preds.T:
            count = Counter(pred)
            total = sum(count.values())
            prob = [count.get(0, 0) / total, count.get(1, 0) / total]
            probs.append(prob)

        return np.array(probs)