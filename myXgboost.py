'''
手写xgboost模型
1、循环增加一颗CART树
2、用贪婪算法构建树，包含用损失函数一阶和二阶导数信息计算叶子的最优权重wj和节点分裂的最优增益Gain
3、用构建好的树迭代优化函数空间：y_hat(t) = y_hat(t-1) + ft(xi)
4、重新更新计算损失函数的一阶和二阶导数
5、重复第(1)步，直到生成K颗树
'''
import pandas as pd
import numpy as np
import copy
import random
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
'''**************************回归树节点**************************'''
class TreeNode(object):
    def __init__(self):
        self.split_feature = None  # 分裂特征
        self.split_value = None  # 分裂切割点
        self.split_gain = None  # 分裂信息增益
        self.internal_value = None
        self.node_index = None
        self.leaf_value = None  # 当前叶子节点的权重值
        self.tree_left = None  # 左子树节点
        self.tree_right = None  # 右子树节点
    '''预测分数：找到当前dataset落在叶子位置的权重值'''
    def calc_predict_value(self, dataset):
        # 如果是叶子节点，则直接返回叶子权重值
        if self.leaf_value is not None:
            return self.leaf_value
        # 如果当前分裂特征值小于分裂切割点，进入左节点预测
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        # 如果当前分裂特征值大于分裂切割点，进入右节点预测
        else:
            return self.tree_right.calc_predict_value(dataset)
    '''递归统计回归树叶子节点数量和叶子节点的父节点数量——用于正则化函数'''
    def state_tree(self, leaves_state, node_state):
        # 如果当前节点不存在左右子树，则说明是叶子节点，叶子节点数量+1
        if not self.tree_left and not self.tree_right:
            leaves_state.append(1)
            return
        # 如果当前节点的左右子树不存在分裂信息增益，则说明是叶子节点的父节点，叶子节点父节点数量+1
        if not self.tree_left.split_gain and not self.tree_right.split_gain:
            node_state.append([self.node_index, self.split_gain])
        # 否则进入左右子树继续尝试统计
        self.tree_left.state_tree(leaves_state, node_state)
        self.tree_right.state_tree(leaves_state, node_state)
        return leaves_state, node_state
    '''递归剪支操作：剪支对象：叶子节点的父节点，剪除叶子节点，将该叶子节点的父节点作为新的叶子节点'''
    def prune_tree(self, prune_node_index):
        # 如果当前节点不存在左右子树，则说明是叶子节点，不用剪支
        if not self.tree_left and not self.tree_right:
            return
        # 如果当前节点的左子树需要剪支
        if self.tree_left.node_index == prune_node_index:
            leaf_value = self.tree_left.internal_value
            self.tree_left = TreeNode()  # 将左子树作为叶子节点
            self.tree_left.node_index = prune_node_index
            self.tree_left.leaf_value = leaf_value
            return
        # 如果当前节点的右子树需要剪支
        elif self.tree_right.node_index == prune_node_index:
            leaf_value = self.tree_right.internal_value
            self.tree_right = TreeNode()  # 将右子树作为叶子节点
            self.tree_right.node_index = prune_node_index
            self.tree_right.leaf_value = leaf_value
            return
        # 否则进入左右子树继续尝试剪支
        self.tree_left.prune_tree(prune_node_index)
        self.tree_right.prune_tree(prune_node_index)
        return
    '''递归输出回归树整体结构'''
    def describe_tree(self):
        # 如果当前节点不存在左右子树，则说明是叶子节点，直接返回叶节点权重
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        # 否则进入左右子树继续尝试输出
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",split_gain:" + str(self.split_gain) + \
                         ",internal_value:" + str(self.internal_value) + \
                         ",node_index:" + str(self.node_index) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure

'''**************************回归树模型**************************'''
class BaseDecisionTree(object):
    def __init__(self, max_depth, num_leaves, min_samples_split, min_samples_leaf, subsample,
                 colsample_bytree, max_bin, min_child_weight, reg_gamma, reg_lambda, random_state):
        self.max_depth = max_depth  # 允许的最大深度
        self.num_leaves = num_leaves  # 允许的叶子数量
        self.min_samples_split = min_samples_split  # 允许分裂后的样本最小数量
        self.min_samples_leaf = min_samples_leaf  # 允许叶节点叶子的最小数量
        self.subsample = subsample  # 模拟随机森林：有放回抽取样本比例
        self.colsample_bytree = colsample_bytree  # 模拟随机森林：无放回选取部分特征比例
        self.max_bin = max_bin
        self.min_child_weight = min_child_weight  # 允许叶子节点的权重最小值
        self.reg_gamma = reg_gamma  # 正则化系数
        self.reg_lambda = reg_lambda  # 正则化系数
        self.random_state = random_state  # 随机数种子
        self.tree = TreeNode()  # 回归树结构
        self.ft_x = None  # 当前回归树的函数预测分数
        self.node_index = 0
        self.feature_importances_ = dict()
        self.obj = 0  # 回归树的误差
    '''训练CART树（带随机采样和剪支处理）'''
    def fit(self, dataset, targets):
        dataset_copy = copy.deepcopy(dataset).reset_index(drop=True)
        targets_copy = copy.deepcopy(targets).reset_index(drop=True)
        # 设置随机种子
        if self.random_state:
            random.seed(self.random_state)
        # 借鉴随机森林思想：随机有放回抽样m个子样本（行抽样）
        if self.subsample < 1.0:
            subset_index = np.random.choice(range(len(targets)), int(self.subsample * len(targets)), replace=True)
            # subset_index = random.sample(range(len(targets)), int(self.subsample * len(targets)))
            dataset_copy = dataset_copy.iloc[subset_index, :].reset_index(drop=True)
            targets_copy = targets_copy.iloc[subset_index, :].reset_index(drop=True)
        # 借鉴随机森林思想：随机无放回抽样m个子特征（列抽样）
        if self.colsample_bytree < 1.0:
            subcol_index = random.sample(list(dataset_copy.columns), int(self.colsample_bytree * len(dataset_copy.columns)))
            dataset_copy = dataset_copy[subcol_index]
        # 递归构建回归树
        self.tree = self._fit(dataset_copy, targets_copy, depth=0)
        # 当前回归树的函数预测分数
        self.ft_x = dataset.apply(lambda x: self.predict(x), axis=1)
        # 递归统计回归树叶子节点数量和叶子节点的父节点数量——用于正则化函数
        leaves_state, node_state = self.tree.state_tree(leaves_state=[], node_state=[])
        # 如果叶子数量大于允许的叶子数量，则进入树剪支操作
        while sum(leaves_state) > self.num_leaves:
            node_state = sorted(node_state, key=lambda x: x[1])
            self.tree.prune_tree(node_state[0][0])  # 树剪支操作
            leaves_state, node_state = self.tree.state_tree(leaves_state=[], node_state=[])
        # 记录回归树的误差
        self.obj = self.calc_one_tree_obj(targets_copy)
        return self
    '''递归构建回归树'''
    def _fit(self, dataset, targets, depth):
        # 如果当前样本数量小于最小样本数量，则停止分裂，直接创建叶节点
        if dataset.__len__() <= self.min_samples_split or targets['hess'].sum() <= self.min_child_weight:
            tree = TreeNode()
            tree.leaf_value = self.calc_leaf_value(targets)
            return tree
        # 如果当前树深度小于允许的最大深度，则尝试继续分裂节点
        if depth < self.max_depth:
            # 选择使得当前节点分裂最好的特征和分割点
            best_split_feature, best_split_value, best_split_gain, best_internal_value = \
                self.choose_best_feature(dataset, targets)
            # 根据特征和分割点二分化dataset和targets
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)
            tree = TreeNode()
            # 如果左子树或右子树的样本量小于允许的样本最小值，则停止分裂，直接创建叶节点
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf:
                tree.leaf_value = self.calc_leaf_value(targets)
                return tree
            # 否则根据split_feature和split_value分裂当前节点，创建左右子树
            else:
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1
                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.split_gain = best_split_gain
                tree.internal_value = best_internal_value
                tree.node_index = self.node_index
                self.node_index += 1
                # 递归构建左右子树
                tree.tree_left = self._fit(left_dataset, left_targets, depth + 1)
                tree.tree_right = self._fit(right_dataset, right_targets, depth + 1)
                return tree
        # 当前树深度超过允许的最大深度，则停止分裂，直接创建叶节
        else:
            tree = TreeNode()
            tree.leaf_value = self.calc_leaf_value(targets)
            return tree
    '''选择使得节点分裂最好的特征'''
    def choose_best_feature(self, dataset, targets):
        best_split_gain, best_split_feature, best_split_value = float('-inf'), None, None
        for feature in list(dataset.columns):
            # 初始化待选分割点
            if dataset[feature].unique().__len__() <= 100:
                unique_values = dataset[feature].unique()
            else:
                # 计算特征值的不同分位数作为待选分割点
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, self.max_bin)])
            # 遍历待选分割点
            for split_value in unique_values:
                # 二分化分裂targets数据集
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_split_gain(left_targets, right_targets)
                if split_gain > best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        best_internal_value = self.calc_leaf_value(targets)
        return best_split_feature, best_split_value, best_split_gain, best_internal_value
    '''计算叶节点的权重值(gradient加权)：由损失函数的一阶和二阶偏导数计算出来'''
    def calc_leaf_value(self, targets):
        Gj, Hj = targets['grad'].sum(), targets['hess'].sum()
        Wj = -Gj / (Hj + self.reg_lambda)
        return Wj
    '''计算节点分裂后的增益(gradient加权)：由损失函数的一阶和二阶偏导数计算出来'''
    def calc_split_gain(self, left_targets, right_targets):
        GL = left_targets['grad'].sum()
        HL = left_targets['hess'].sum()
        GR = right_targets['grad'].sum()
        HR = right_targets['hess'].sum()
        split_gain = 0.5 * (GL ** 2 / (HL + self.reg_lambda) +
                            GR ** 2 / (HR + self.reg_lambda) -
                            (GL + GR) ** 2 / (HL + HR + self.reg_lambda)) - self.reg_gamma
        return split_gain
    '''计算回归树的误差：由损失函数的一阶和二阶偏导数计算出来'''
    def calc_one_tree_obj(self, targets):
        leaves_state, _ = self.tree.state_tree(leaves_state=[], node_state=[])
        obj = -0.5 * (targets['grad'].sum() ** 2) / (
                targets['hess'].sum() + self.reg_lambda) + self.reg_gamma * len(
            leaves_state)
        return obj
    '''根据特征和分割点二分化dataset和targets'''
    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets
    '''预测dataset的分数，找到当前dataset落在叶子位置的权重值'''
    def predict(self, dataset):
        return self.tree.calc_predict_value(dataset)
    '''输出回归树的整体结构'''
    def print_tree(self):
        return self.tree.describe_tree()

'''**************************经典损失函数**************************'''
'''
损失函数：R2误差
L = 1/2 * (yhat - y)**2
'''
class SquareLoss(object):
    # 一阶偏导数
    def grad(self, targets):
        gi = targets['pred'] - targets['label']
        return gi
    # 二阶偏导数
    def hess(self, targets):
        hi = 1
        return hi
'''
损失函数：Logistic误差
L = -(y * log yhat + (1 - y) * log(1 - yhat))
label = [0, 1]
'''
class LogisticLoss(object):
    # 一阶偏导数
    def grad(self, targets):
        yhat, y = sigmoid(x=targets['pred']), targets['label']
        gi = (1 - y) / (1 - yhat) - y / yhat
        return gi
    # 二阶偏导数
    def hess(self, targets):
        yhat, y = sigmoid(x=targets['pred']), targets['label']
        hi = (1 - y) / ((1 - yhat) ** 2) + y / yhat ** 2
        return hi
'''**************************XGBoost算法**************************'''
class myXGBClassifier(object):
    def __init__(self, n_estimators=100, max_depth=-1, num_leaves=-1, learning_rate=0.1, min_samples_split=2,
                 min_samples_leaf=1, subsample=1., colsample_bytree=1., max_bin=225, min_child_weight=1.,
                 reg_gamma=0., reg_lambda=0., loss='logistic', random_state=None):
        self.n_estimators = n_estimators  # 回归树的数量
        self.max_depth = max_depth if max_depth != -1 else float('inf')  # 允许的最大深度
        self.num_leaves = num_leaves if num_leaves != -1 else float('inf')  # 允许的最大叶子数量
        self.learning_rate = learning_rate  # 梯度下降学习率
        self.min_samples_split = min_samples_split  # 允许分裂后的样本最小数量
        self.min_samples_leaf = min_samples_leaf  # 允许叶节点叶子的最小数量
        self.subsample = subsample  # 模拟随机森林：有放回抽取样本比例
        self.colsample_bytree = colsample_bytree  # 模拟随机森林：无放回选取部分特征比例
        self.max_bin = max_bin
        self.min_child_weight = min_child_weight  # 允许叶子节点的权重最小值
        self.reg_gamma = reg_gamma  # 正则化系数
        self.reg_lambda = reg_lambda  # 正则化系数
        self.loss = loss  # 损失函数类型
        self.random_state = random_state  # 随机数种子
        self.pred_0 = 0
        self.trees = dict()  # 回归树集合
        self.feature_importances_ = dict()
    '''训练XGBClassifier'''
    def fit(self, dataset, targets):
        # 初始化损失函数
        if self.loss == 'logistic':
            self.loss = LogisticLoss()  # 逻辑损失函数
        elif self.loss == 'squareloss':
            self.loss = SquareLoss()  # R2损失函数
        else:
            raise ValueError("The loss function must be 'logistic' or 'squareloss'!")
        targets = targets.to_frame(name='label')
        # 二分类标签检测
        if targets['label'].unique().__len__() != 2:
            raise ValueError("There must be two class for targets!")
        # 数值类型检测
        if len([x for x in list(dataset.columns) if dataset[x].dtype in ['int32', 'float32', 'int64', 'float64']]) \
                != len(list(dataset.columns)):
            raise ValueError("The features dtype must be int or float!")
        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(max(self.n_estimators, len(targets))), self.n_estimators)
        # 初始化没有加入回归树的状态
        # mean = 1.0 * sum(targets['label']) / len(targets['label'])
        # self.pred_0 = 0.5 * log((1 + mean) / (1 - mean))
        targets['pred'] = self.pred_0
        targets['grad'] = targets.apply(self.loss.grad, axis=1)
        targets['hess'] = targets.apply(self.loss.hess, axis=1)
        # 贪心策略，加入n_estimators颗回归树
        for stage in range(self.n_estimators):
            print('加入第{}颗树'.format(stage + 1).center(80, '='))
            # 加入1颗回归树
            tree = BaseDecisionTree(self.max_depth, self.num_leaves, self.min_samples_split, self.min_samples_leaf,
                                    self.subsample, self.colsample_bytree, self.max_bin, self.min_child_weight,
                                    self.reg_gamma, self.reg_lambda, random_state_stages[stage])
            # 用贪婪算法构建树，包含用损失函数一阶和二阶导数信息计算叶子权重
            tree = tree.fit(dataset, targets)
            self.trees[stage] = tree  # 将训练好的回归树加入树集合
            # 用构建好的树迭代优化函数空间：y(t) = y(t-1) + a * ft(x)
            targets['pred'] = targets['pred'] + self.learning_rate * tree.ft_x
            # 重新计算损失函数的一阶偏导数(gradient加权)
            targets['grad'] = targets.apply(self.loss.grad, axis=1)
            # 重新计算损失函数的二阶偏导数(gradient加权)
            targets['hess'] = targets.apply(self.loss.hess, axis=1)
            # 更新特征重要程度表
            for key, value in tree.feature_importances_.items():
                self.feature_importances_[key] = self.feature_importances_.get(key, 0) + 1
        return self
    '''预测测试集类别'''
    def predict(self, dataset):
        res = []
        for i, row in dataset.iterrows():
            fi_value = 0
            # 遍历每一颗树，累加每个样本对应的叶子权重分数
            for stage, tree in self.trees.items():
                fi_value += tree.predict(row)
            # 对累加分数进行非线性映射，通过阈值判定预测类别
            pred_label = 1 if sigmoid(x=fi_value) >= 0.5 else 0
            res.append(pred_label)
        return np.array(res)
if __name__ == '__main__':
    df = pd.read_csv('data/xg.csv')
    train_count = int(0.7 * len(df))
    train_data, train_label = df.iloc[:train_count, :-1], df.iloc[:train_count, -1]
    test_data, test_label = df.iloc[train_count:, :-1], df.iloc[train_count:, -1]
    xgb = myXGBClassifier(n_estimators=30,
                        max_depth=6,
                        num_leaves=30,
                        learning_rate=0.1,
                        min_samples_split=40,
                        min_samples_leaf=10,
                        subsample=0.6,
                        colsample_bytree=0.8,
                        max_bin=150,
                        min_child_weight=1,
                        reg_gamma=0.1,
                        reg_lambda=0.3,
                        loss='logistic',
                        random_state=66)
    xgb = xgb.fit(dataset=train_data, targets=train_label)
    # 测试模型准确率
    from sklearn.metrics import accuracy_score, roc_auc_score
    predictions = xgb.predict(dataset=test_data)
    print('myXgboost模型准确度为：{:.2%}，ROC曲线下面积为：{:.2%}'.format(
        accuracy_score(test_label, predictions),
        roc_auc_score(test_label, predictions)
    ))
    # 对比封装好的xgboost模型
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=30).fit(train_data, train_label)
    predictions = model.predict(test_data)
    print('Xgboost模型准确度为：{:.2%}，ROC曲线下面积为：{:.2%}'.format(
        accuracy_score(test_label, predictions),
        roc_auc_score(test_label, predictions)
    ))
