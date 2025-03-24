from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    return xgb_clf

def train_lightgbm(X_train, y_train):
    lgb_clf = lgb.LGBMClassifier()
    lgb_clf.fit(X_train, y_train)
    return lgb_clf
