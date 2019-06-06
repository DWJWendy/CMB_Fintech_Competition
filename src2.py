#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- encoding: utf-8 -*-
"""
@File    : src2.ipynb
@Time    : 2019-04-29 11:22
@Author  : 邓文君
@Email   : dengwenjun818@gmail.com
@Software: jupyter
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib

data_path = 'data/FT_Camp_2/'
train_path = data_path+'train.csv'
trx_cod_path = data_path+'trx_cod.csv'
sz_detail_path = data_path + 'sz_detail.csv'
cust_bas_inf_path = data_path+'cust_bas_inf.csv'
pred_users_path = data_path + 'pred_users.csv'
res_path = 'res/'


def read_file(path):
    """
    path: 读取数据的路径
    """
    return pd.read_csv(path, sep=',')


def write_result(item, path=res_path+'results.csv'):
    """
    item: pred_users 格式为pandas
    path: 默认
    将预测结果写入结果表
    """
    item.to_csv(path, sep=',', encoding='utf-8')
    
# 获取cat1、cat2、g2_cod的取值列表
cat1_list = list(read_file(trx_cod_path)['cat1'].unique())
cat2_list = list(read_file(trx_cod_path)['cat2'].unique())
g2_list = list(read_file(sz_detail_path)['g2_cod'].unique())
cat1m_list = list(map(lambda x:x+'M', cat1_list))
cat2m_list =  list(map(lambda x:x+'M', cat2_list))
g2m_list =  list(map(lambda x:str(x)+'M', g2_list))

class DataPrecess(object):
    """
    数据预处理:特征映射、训练集和测试构建
    """
    def __init__(self, train_path, trx_cod_path, cust_bas_inf_path, sz_detail_path, pred_users_path):
        """
        读取所需的数据
        """
        self.raw_train = read_file(train_path)
        self.trx_code_data = read_file(trx_cod_path)
        self.cust_bas_inf_data = read_file(cust_bas_inf_path)
        self.sz_detail_data = read_file(sz_detail_path)
        self.pred_data = read_file(pred_users_path)


    def feature_map(self, items, k='train'):
        """
        k='pred' or 'train'
        基于交易和用户的两类特征进行融合
        """
        print('正在进行'+ k +'用户数据和交易数据的融合表示...')
        new_items = self.error_value_deal(pd.merge(items, self.cust_bas_inf_data, how='left', on=['id']))
        transaction_data = pd.merge(new_items, self.sz_detail_data, how='left', on=['id'])
        transaction_data = pd.merge(transaction_data, self.trx_code_data, how='left', on=['sz_id'])
        write_result(transaction_data, path=res_path+k+'_transaction_data.csv')
        write_result(new_items, path=res_path+k+'_user_data.csv')
        print(k+'训练集用户和交易数据的融合完成...')
        return new_items, transaction_data
    
    def error_value_deal(self, user_data):
        """
        user_data:用户数据
        对数据中性别错误值进行更改,按照最多性别进行赋值,统计发现男性更多，对数据中对年龄、aum227和aum306使用均值替代
        """
        # 使用特征的众数对age和gender进行填充
        imp1 = SimpleImputer(missing_values='\\N', strategy='most_frequent')  
        user_data['age'] = imp1.fit_transform(np.array(user_data['age']).reshape(-1, 1))
        user_data['gender'] = imp1.fit_transform(np.array(user_data['gender']).reshape(-1, 1))
        # 使用均值对aum227和aum306填充 
        aum227,aum306 = list(user_data['aum227']),list(user_data['aum306'])
        if "\\N" in aum227:
            aum227_mean = np.array(list(map(lambda x:float(x) if x!='\\N' else 0, aum227))).mean()
            user_data['aum227'] = list(map(lambda x:float(x) if x!='\\N' else aum227_mean,aum227))
        if "\\N" in aum306:
            aum306_mean = np.array(list(map(lambda x:float(x) if x!='\\N' else 0, aum306))).mean()
            user_data['aum306'] = list(map(lambda x:float(x) if x!='\\N' else aum306_mean,aum306))
        return user_data

    @staticmethod
    def transacation_map(user, transaction, prt_dt, k):
        """
        k='pred' or 'train'
        transaction:用户交易数据
        目标：抽取用户交易数据，确定给定时间到用户交易特征
        """
        print('正在进行'+ k +'交易数据抽取...')
        user_col = list(user.columns)
        new_col = g2_list+cat1_list +cat2_list+ ['sz_amt','expend','income']
        new_data = pd.DataFrame(index=user.index, columns=user_col+new_col)
        new_data[user_col] = user[user_col]
        transaction = transaction[transaction['prt_dt']<= prt_dt]
        for i in new_data.index:
            if i%2000==0:
                print(i)
            uid = new_data.loc[i, 'id']
            tmp = transaction[(transaction['id']==uid)][['cat1','cat2','g2_cod','rmb_amt']]
            sz_amt = sum(tmp['rmb_amt'])
            expend = sum(tmp[tmp['rmb_amt']<0]['rmb_amt'])
            new_data.loc[i, 'sz_amt'] = sz_amt  # 计算在prt_dt时间之前的资金往来记录
            new_data.loc[i, 'expend'] = expend  # 计算在prt_dt时间之前的资金支出数
            new_data.loc[i, 'income'] = sz_amt - expend
            # cat1\cat2\g2_cod对应的频数
            try:
                cat1_group= tmp.groupby(['cat1']).size()
                for c1 in cat1_group.index:
                    new_data.loc[i, c1] = cat1_group[c1]
            except: pass
            try:
                cat2_group= tmp.groupby(['cat2']).size()
                for c2 in cat2_group.index:
                    new_data.loc[i, c2] = cat2_group[c2]          
            except: pass
            try:
                g2_cod_group = tmp.groupby(['g2_cod']).size()
                for g in g2_cod_group.index:
                    new_data.loc[i, g] = g2_cod_group[g]
            except: pass
            # cat1\cat2\g2_cod对应的金额均值
            try:
                cat1_group= tmp['rmb_amt'].groupby(tmp['cat1']).mean()
                for c1 in cat1_group.index:
                    new_data.loc[i, c1+'M'] = cat1_group[c1]
            except: pass
            try:
                cat2_group= tmp['rmb_amt'].groupby(tmp['cat2']).mean()
                for c2 in cat2_group.index:
                    new_data.loc[i, c2+'M'] = cat2_group[c2]
            except: pass
            try:
                g2_cod_group = tmp['rmb_amt'].groupby(tmp['g2_cod']).mean()
                for g in g2_cod_group.index:
                    new_data.loc[i, str(g)+'M'] = g2_cod_group[g]
            except: pass   
        new_data = new_data.fillna(0)
        print('完成'+k+'交易数据抽取...')
        write_result(new_data, path=res_path+ k +'_feature_data.csv')
        return new_data

    @staticmethod
    def one_hot(items, k):
        """
        对gender进行onehot处理, 由于在之前做过错误值更替处理，所以gender只有M和F两个取值
        """
        print('正在进行'+k+'数据的one-hot表征...')
        ohe = OneHotEncoder()
        ohe.fit([['M'],['F']])
        one_hot_data = list(map(lambda x: [x], list(items['gender'])))
        items['gender'] = list(ohe.transform(one_hot_data).toarray())
        joblib.dump(items, res_path+k+'_feature_ohe_data_1.pkl')
        print('完成'+k+'数据的one-hot表征...')
        return items
    
    @staticmethod
    def form_vec(items, k):
        """
        items: 将融合的数据用向量表示形成特征向量
        k='pred' or 'train'
        """
        print('正在进行'+k+'数据表示成特征向量...')
        if k=='pred':
            col = list(items.columns)[4:]
            col.remove('aum227')
            items['vec'] = items.apply(lambda x: list(x[col]), axis = 1)
            items['gender'] = list(map(lambda x: list(x), list(items['gender'])))
            items['vec'] = items['vec']+ items['gender']
            joblib.dump(items[['id', 'vec']], res_path+k+'_feature_vec_1.pkl')
            print(k+'数据表示成特征向量完成...')
            return items[['id', 'vec']]
            
        elif k=='train':
            col = list(items.columns)[5:]
            col.remove('aum306')
            items['vec'] = items.apply(lambda x: list(x[col]), axis = 1)
            items['gender'] = list(map(lambda x: list(x), list(items['gender'])))
            items['vec'] = items['vec']+ items['gender']
            joblib.dump(items[['id', 'vec', 'click_w228']], res_path+k+'_feature_vec_1.pkl')
            print(k+'数据表示成特征向量完成...')
            return items[['id', 'vec', 'click_w228']]
            
        
        
    
    @staticmethod
    def normalization(items, k):
        """
        items: 将形成的特征向量数据
        对每个用户对特征向量进行归一化处理，使用min_max归一化对方式
        """
        print('正在进行'+k+'数据归一化处理...')
        min_max_scaler = MinMaxScaler()
        items['vec'] = list(min_max_scaler.fit_transform(list(items['vec'])))
        joblib.dump(items, res_path+k+"_vec_normal_1.pkl")
        print('完成'+k+'数据归一化...')
        return items
    
    def pred_data_run(self, pred_k='pred', pred_date='2019-03-05'):
        """
        pred数据预处理
        """
        """1. 计算pred的过程"""
        print("计算pred数据中...")
        user, transaction = self.feature_map(dataprecess.pred_data, pred_k)
        pred_feature_data = self.transacation_map(user, transaction, pred_date, pred_k)
        pred_feature_data = read_file('res/pred_feature_data_1.csv')
        pred_feature_ohe_data = self.one_hot(pred_feature_data,pred_k)
        pred_feature_vec =self.form_vec(pred_feature_ohe_data, pred_k)
        pred_vec_normal = self.normalization(pred_feature_vec, pred_k)
    
    def train_data_run(self, train_k='train', train_date='2019-02-27'):
        """
        train数据预处理
        """
        """1. 计算train的过程"""
        print("计算train数据过程...")
        user, transaction = self.feature_map(dataprecess.raw_train, train_k)
        train_feature_data =self.transacation_map(user, transaction, train_date, train_k)
        train_feature_data = read_file('res/train_feature_data_1.csv')
        train_feature_ohe_data = self.one_hot(train_feature_data,train_k)
        train_feature_vec =self.form_vec(train_feature_ohe_data, train_k)
        train_vec_normal = self.normalization(train_feature_vec, train_k)
       
    
if __name__== "__main__":
    dataprecess = DataPrecess(train_path, trx_cod_path, cust_bas_inf_path,sz_detail_path,pred_users_path)
    dataprecess.pred_data_run() # 1. 计算pred的过程
    dataprecess.train_data_run() # 2. 计算train的过程 


# In[ ]:


# 基于sklean接口构建xgboost模型，模型构建完成且可运行
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import ShuffleSplit

# 设置xgboost的多个参数值，以便选择最优参数
param_grid = dict(max_depth = [5,6,7],learning_rate = [0.02,0.03,0.04,0.05],n_estimators = [500,600,700])
res_path = 'res/'


class ClassiferModel(object):
    def __init__(self, train_data, pred_data, oversampling='s'): 
        self.train_vec = np.array(list(train_data['vec']))
        self.train_label = np.array(list(train_data['click_w228']))
        self.pred_data = pred_data
        self.oversampling = oversampling
        self.train_x, self.train_y = self.data_prepare()

    def do_fit(self, feature):
        """
        模型拟合, 使用GridSearchCV进行参数选择
        """
        print("正在进行xgboost训练...")
        if feature:# 进行特征选择
            model = joblib.load(res_path+ self.oversampling+'_feature_select_'+str(num_boost_round)+'.pkl')
            train_x,train_y = model.transform(self.train_x), np.array(self.train_y)
        else: # 不进行特征选择
            train_x, train_y = np.array(self.train_x), np.array(self.train_y)
        xgb_model = XGBClassifier(n_jobs=5)
        cv_split = ShuffleSplit(n_splits=4, train_size=0.7, test_size=0.3) # 划分测试集以及确定交叉验证分片数
        grid = GridSearchCV(xgb_model, param_grid, cv=cv_split, scoring='roc_auc')
        grid.fit(train_x, train_y)
        best_xgb_model = grid.best_estimator_  # 得到最好的迭代器
        joblib.dump(best_xgb_model,res_path + self.oversampling + '_xgboost_'+ str(feature) + '.pkl')
        print("xgboost训练完成...")
    
    def data_prepare(self):
        """
        数据准备，由于数据集存在不均衡的现象[点击数：未点击数=1:8],所以对训练集数据进行处理，处理方式为对训练集中对点击数据进行随机过采样
        """
        print('正在进行数据集准备...')
        if self.oversampling=='s':
            ros = RandomOverSampler(random_state=0)
            new_train_vec, new_train_label = ros.fit_sample(self.train_vec, self.train_label) 
        else:
            new_train_vec, new_train_label = self.train_vec, self.train_label
        print('数据集准备完成...')
        return new_train_vec, new_train_label
    
    def feature_select(self):
        """
        基于树模型选择特征
        """
        print("进行特征选择计算...")
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(self.train_x, self.train_y)
        model = SelectFromModel(clf, prefit=True)
        joblib.dump(model,res_path+ self.oversampling+'_feature_select_'+'.pkl')
        print("特征选择完成...")
        return model
    
    def predict(self, feature):
        print("正在模型预测...")
        predict_res = pd.DataFrame(index=list(self.pred_data.index), columns=['id', 'score'])
        dtest = np.array(list(self.pred_data['vec']))
        best_xgb_model = joblib.load(res_path + self.oversampling+'_xgboost_'+str(feature)+'.pkl')
        ypred = list(map(lambda x:x[1], best_xgb_model.predict_proba(dtest)))
        predict_res['id'] = self.pred_data['id']
        predict_res['score']= ypred
        predict_res.to_csv(res_path+self.oversampling+'_results_'+str(feature)+'.csv', sep=',', encoding='utf-8')
        print("模型预测完成...")
    
    def run(self,feature=True):
        print('现在进行模型的训练和预测...')
        if feature:
            model = self.feature_select()
        self.do_fit(feature)
        self.predict(feature)
        
if __name__ == "__main__":
    train_data = joblib.load(res_path+'train_vec_normal.pkl')
    pred_data = joblib.load(res_path+'pred_vec_normal.pkl')
    classifier = ClassiferModel(train_data, pred_data, 's')
    classifier.run(feature=False)


# In[ ]:


# 【由于基于sklearn的xgboost模型需要较长时间进行模型训练和参数选择】
# 基于原生接口构建xgboost模型
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


num_boost_round= 200
res_path = 'res/'
# 模型参数设置
params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':6,
    'lambda':1,
    'subsample':0.85,
    'colsample_bytree':0.85,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0}


def write_result(item, path=res_path+'results.csv'):
    """
    item: pred_users 格式为pandas
    path: 默认
    将预测结果写入结果表
    """
    item.to_csv(path, sep=',', encoding='utf-8')
    

class ClassiferModel(object):
    def __init__(self, train_data, pred_data, oversampling='s'): 
        self.train_vec = np.array(list(train_data['vec']))
        self.train_label = np.array(list(train_data['click_w228']))
        self.pred_data = pred_data
        self.oversampling = oversampling
        self.train_x, self.test_x, self.train_y, self.test_y = self.data_prepare()
    
    def data_prepare(self):
        """
        由于数据集存在不均衡的现象[点击数：未点击数=1:8],所以对训练集数据进行处理，处理方式为对训练集中对点击数据进行随机过采样
        划分训练集和测试集，测试集和训练集分别占比3:7 且按照lable的比例进行抽取
        """
        print('划分测试集和训练集...')
        if self.oversampling=='s':
            ros = RandomOverSampler(random_state=0)
            new_train_vec, new_train_label = ros.fit_sample(self.train_vec, self.train_label)
            train_x, test_x, train_y, test_y = train_test_split(new_train_vec, new_train_label, test_size=0.3)  
        else:
            train_x, test_x, train_y, test_y = train_test_split(self.train_vec, self.train_label, test_size=0.3, 
                                                                stratify=list(self.train_label))
        return train_x, test_x, train_y, test_y
    
    def do_fit(self, feature):
        """
        模型拟合
        """
        print("正在进行xgboost训练...")
        if feature:# 进行特征选择
            model = joblib.load(res_path + 'feature_select.pkl')
            train_x,train_y = model.transform(self.train_x), np.array(self.train_y)
        else: # 不进行特征选择
            train_x, train_y= np.array(self.train_x), np.array(self.train_y)
        dtrain = xgb.DMatrix(train_x, label=train_y)
        watchlist = [(dtrain, 'train')]
        bst = xgb.train(params,dtrain,num_boost_round, evals=watchlist)
        bst.save_model(res_path + 'xgboost_5.model')

    def auc_predict(self,feature):
        """
        预测结果的指标计算
        """
        if feature:
            model = joblib.load(res_path + 'feature_select.pkl')
            test_x = model.transform(self.test_x)
        else:
            test_x = np.array(self.test_x)
        dtest = xgb.DMatrix(test_x)
        bst = xgb.Booster({'nthread': 4})
        bst.load_model(res_path + 'xgboost_5.model')
        predict_y = bst.predict(dtest)
        test_y, predict_y = list(self.test_y), list(predict_y)
        fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y)
        return metrics.auc(fpr, tpr)
    
    def feature_select(self):
        """
        基于树模型选择特征
        """
        print("进行特征选择计算...")
        clf = ExtraTreesClassifier(n_estimators=100)
        clf = clf.fit(self.train_x, self.train_y)
        model = SelectFromModel(clf, prefit=True)
        joblib.dump(model, res_path + 'feature_select.pkl')
        print("完成特征选择计算...")
        return model
    
    def predict(self, feature):
        print("正在模型预测阶段...")
        predict_res = pd.DataFrame(index=list(self.pred_data.index), columns=['id', 'score'])
        test_x = np.array(list(self.pred_data['vec']))
        bst = xgb.Booster({'nthread': 4})
        bst.load_model(res_path + 'xgboost_5.model')
        dtest = xgb.DMatrix(test_x)
        ypred = bst.predict(dtest)
        predict_res = self.pred_data[['id']]
        predict_res['score']=ypred
        write_result(predict_res, path=res_path + 'results_5.csv')
        return predict_res
    
    def run(self,feature=True):
        print('现在进行模型的训练和预测...')
        if feature:
            model = self.feature_select()
        self.do_fit(feature)
        print(self.auc_predict(feature))
        self.predict(feature)
        
if __name__ == "__main__":
    train_data = joblib.load(res_path+'train_vec_normal_1.pkl')
    pred_data = joblib.load(res_path+'pred_vec_normal_1.pkl')
    classifier = ClassiferModel(train_data, pred_data, 's')
    classifier.run(feature=False)


# In[ ]:


# 多模型对比实验
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split

# 模型初始化
lr_model = LogisticRegression()
gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5, subsample=0.8)
rf_model = RandomForestClassifier()
bayes = BernoulliNB()
nn_model = MLPClassifier()
svm_model = SVC(kernel='rbf', gamma='scale')
xgboost_model = XGBClassifier()
models = {'lr':lr_model, 'gb':gb_model, 'rf':rf_model, 'bayes':bayes, 'nn':nn_model, 'svm':svm_model}
res_path ='res/'

def read_file(path):
    """
    path: 读取数据的路径
    """
    return pd.read_csv(path, sep=',')

def write_result(item, path=res_path+'results.csv'):
    """
    item: pred_users 格式为pandas
    path: 默认
    将预测结果写入结果表
    """
    item.to_csv(path, sep=',', encoding='utf-8')
    
class ClassiferModel(object):
    def __init__(self, train_data, pred_data,oversampling='s'): 
        self.train_vec = np.array(train_data['vec'])
        self.train_label = np.array(train_data['click_w228'])
        self.pred_data = pred_data
        self.oversampling = oversampling
        self.train_x, self.test_x, self.train_y, self.test_y = self.data_prepare()
        
    def data_prepare(self):
        """
        由于数据集存在不均衡的现象[点击数：未点击数=1:8],所以对训练集数据进行处理，处理方式为对训练集中对点击数据进行随机过采样
        划分训练集和测试集，测试集和训练集分别占比3:7 且按照lable的比例进行抽取
        """
        print('划分测试集和训练集...')
        if self.oversampling=='s':
            ros = RandomOverSampler(random_state=0)
            new_train_vec, new_train_label = ros.fit_sample(self.train_vec, self.train_label)
            train_x, test_x, train_y, test_y = train_test_split(new_train_vec, new_train_label, test_size=0.3)  
        else:
            train_x, test_x, train_y, test_y = train_test_split(self.train_vec, self.train_label, test_size=0.3, 
                                                                stratify=list(self.train_label))
        return train_x, test_x, train_y, test_y
    
    @staticmethod
    def do_fit(train_x, train_y, test_x, model, k=1):
        """
        模型拟合
        """
        model.fit(train_x, train_y)
        if k==1:
            predict_y = model.predict(test_x)
        elif k==2:
            predict_y = model.predict_proba(test_x)
        return predict_y
    
    @staticmethod
    def auc_predict(test_y, predict_y):
        """
        预测结果的指标计算
        """
        test_y, predict_y = list(test_y), list(predict_y)
        fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y)
        return metrics.auc(fpr, tpr)
    
    def model_select(self):
        res = {}
        for k, model in models.items():
            predict_y = self.do_fit(self.train_x, self.train_y, self.test_x, model)
            res[k] = self.auc_predict(self.test_y, predict_y)
        sort_res = sorted(res.items(), key=lambda x:x[1], reverse = True)
        return sort_res
    
    def predict(self):
        predict_res = pd.DataFrame(index=list(self.pred_data.index), columns=['id', 'score'])
        best_model = models[self.model_select()[0][0]]
        predict_y = self.do_fit(self.train_x, self.train_y, np.array(self.pred_data['vec']), best_model, k=2)
        predict_res = self.pred_data[['id']]
        predict_res['score'] = list(map(lambda x:x[1], predict_y))
        write_result(predict_res, path=res_path+'results.csv')
        return predict_res
    
    
if __name__ == "__main__":
    train_data = joblib.load(res_path+'train_vec_normal.pkl')
    pred_data = joblib.load(res_path+'pred_vec_normal.pkl')
    classifier = ClassiferModel(train_data, pred_data)
    print(classifier.predict())


# In[ ]:


import xlab
if __name__ =="__main__":
    xlab.ftcamp.submit('res/results_5.csv')
    xlab.ftcamp.get_submit_hist() 


# In[ ]:




