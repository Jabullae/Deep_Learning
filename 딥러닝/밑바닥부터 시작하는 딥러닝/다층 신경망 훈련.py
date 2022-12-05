from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve, classification_report

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer


# 평가지표 분류
def get_clf_eval(y_test, pred = None, pred_proba = None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    print('정확도 : {:.4f}, 정밀도 : {:.4f}, 재현율 : {:.4f}, F1 : {:.4f}, AUC : {:.4f}'.
          format(accuracy, precision, recall, f1, roc_auc))



cancer = load_breast_cancer()      # 사이킷런에서 유방암 데이터 가져오기
x=cancer.data                # x축에 input 데이터 나열
y=cancer.target              # y축에 타겟 데이터 나열
x_train_all, x_test, y_train_all, y_test = \
  train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)  # 훈련 데이터와 테스트 데이터 분류
x_train, x_val, y_train, y_val = \
  train_test_split(x_train_all,y_train_all,stratify=y_train_all, \
                   test_size=0.2,random_state=42)  # 훈련 데이터와 검증 데이터 분류

scaler = StandardScaler()   # 객체 만들기
scaler.fit(x_train)     # 변환 규칙을 익히기
x_train_scaled = scaler.transform(x_train)  # 데이터를 표준화 전처리
x_val_scaled = scaler.transform(x_val)      # 데이터를 표준화 전처리

mlp = MLPClassifier(hidden_layer_sizes=(10, ), activation='logistic', \
                    solver='sgd', alpha=0.01, batch_size=32, \
                    learning_rate_init=0.1, max_iter=1000)  # 객체 생성

mlp2 = mlp.fit(x_train_scaled, y_train)    # 훈련하기
mlp.score(x_val_scaled, y_val)      # 정확도 평가
pred = mlp.predict(x_val_scaled)
pred_proba = mlp.predict_proba(x_val_scaled)[:, 1]


# 모델 평가
get_clf_eval(y_val, pred, pred_proba)

! pip install mglearn
import mglearn
import matplotlib.pyplot as plt

x_val_scaled[ :, 0].shape
x_val_scaled[:, 1].shape
y_val.shape

plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("hidden unit")
plt.ylabel("input property")
plt.colorbar()
