import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, classification_report, roc_auc_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import dill as pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import joblib

# Импорт наборов данных
######################
malicious_dataset = pd.read_csv('C:/TLS_Traffic_Classifier/malicious/malicious_flows_ORIGIN.csv')
malicious_dataset2 = pd.read_csv('C:/TLS_Traffic_Classifier/malicious/malicious_flows.csv')
malicious_dataset11 = pd.read_csv('C:/TLS_Traffic_Classifier/malicious/malicious_flows_win11.csv')
malicious_dataset10 = pd.read_csv('C:/TLS_Traffic_Classifier/malicious/malicious_flows_win10.csv')
neispolz=pd.read_csv('C:/TLS_Traffic_Classifier/malicious/malicious_flows_test.csv')
malicious_dataset7 = pd.read_csv('C:/TLS_Traffic_Classifier/malicious/malicious_flows_win7_16.csv')
###################
benign_dataset = pd.read_csv('C:/TLS_Traffic_Classifier/benign/sample_benign_flows.csv')
benign_dataset2 = pd.read_csv('C:/TLS_Traffic_Classifier/benign/benign_flows.csv')
benign_dataset3 = pd.read_csv('C:/TLS_Traffic_Classifier/benign/benign_flows_normal.csv')
benign_dataset4 = pd.read_csv('C:/TLS_Traffic_Classifier/benign/benign_flows_normal2.csv')

# Объединение наборов данных
all_flows = pd.concat([malicious_dataset, malicious_dataset2, malicious_dataset11, malicious_dataset10, malicious_dataset7, neispolz, benign_dataset, benign_dataset2,benign_dataset3,benign_dataset4]).fillna(0)
#all_flows = pd.concat([benign_dataset2, malicious_dataset2])
#all_flows = pd.concat([malicious_dataset, benign_dataset])

# Проверка наборов данных на наличие столбцов и строк с отсутствующими значениями
missing_values = all_flows.isnull().sum()
overall_percentage = (missing_values/all_flows.isnull().count())

# Уменьшение размера набора данных для сокращения времени, затрачиваемого на обучение моделей
reduced_dataset = all_flows.sample(40000, replace=True)
reduced_dataset.fillna(reduced_dataset.mean(), inplace=True)

# Выделение независимых и зависимых переменных для обучающего набора данных
reduced_y = reduced_dataset['isMalware']
reduced_x = reduced_dataset.drop(['isMalware'], axis=1);

# Разделение наборов данных на обучающие и тестовые данные
seed = 42
scoring = 'accuracy'
x_train, x_test, y_train, y_test = model_selection.train_test_split(reduced_x, reduced_y, test_size=0.2, random_state=seed) #random_state=24

# x_test.head().to_csv('x_test.csv', sep=',')
# y_test.head().to_csv('y_test.csv', sep=',')
#print(reduced_dataset.isna().any())

xx = reduced_dataset.groupby("isMalware")["entropy"].mean()
kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
plt.figure(figsize=(10,7), dpi= 80)
xx.plot(kind="bar")
plt.legend()
plt.savefig('C:/diagrams/entropy.png')
plt.show()


#Определение гиперпараметров Случайного леса
# val_scores = []
# for k in range(1, 10):
# 	rf_clf = RandomForestClassifier(max_depth=k)
# 	rf_clf.fit(x_train,y_train)
# 	predicted = rf_clf.predict(x_test)
# 	acc_score = accuracy_score(predicted, y_test)
# 	val_scores.append(acc_score)
# plt.plot(list(range(1, 10)), val_scores)
# plt.xticks(list(range(1, 10)))
# plt.xlabel('')
# plt.ylabel('accuracy_score')
# plt.savefig('C:/diagrams/Допустимость значений Случайного леса.png')
# plt.show()




############################# СЛУЧАЙНЫЙ ЛЕС  ###########################################################################################
##Обучение классификатора Случайного леса
#rf_clf = RandomForestClassifier(random_state=1, max_depth=5, min_samples_leaf=5)
#rf_clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, max_depth=2, max_features = 'auto', n_estimators=100)
#max_depth=2
rf_clf = RandomForestClassifier(max_depth=100)
rf_clf.fit(x_train, y_train)
rf_prediction = rf_clf.predict(x_test)
print(classification_report(y_test, rf_prediction))
print('accuracy: ', accuracy_score(y_test, rf_prediction))
print('precision:', precision_score(y_test, rf_prediction))
print('recall:', recall_score(y_test, rf_prediction))
print('f1-score:', f1_score(y_test, rf_prediction))
conf_m = confusion_matrix(y_test, rf_prediction)
#Вывод схемы несовместимой матрицы ()
sns.heatmap(conf_m, annot=True)
plt.title('Матрица несовместимости Случайного леса')
plt.xlabel('')
plt.ylabel('')
print('Матрица несовместимости для точности набора тестов для  Случайного леса:\n',conf_m)
plt.savefig('C:/diagrams/Матрица несовместимости Случайного леса.png')
plt.show()

############################ ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ  ##############################################################################
# Обучение классификатора Логистической регрессии
#solver='lbfgs', max_iter=100

lm = LogisticRegression()
lm.fit(x_train,y_train)
predictions = lm.predict(x_test)
print(classification_report(y_test, predictions))
conf_m_lr = confusion_matrix(y_test, predictions) # Матрица несовместимости для точности набора тестов
print('Матрица несовместимости для точности набора тестов для Логистической регрессии:\n',conf_m_lr)
sns.heatmap(conf_m_lr, annot=True) #Вывод схемы несовместимой матрицы ()
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.savefig('C:/diagrams/Матрица несовместимости ЛР.png')
plt.show()
print('accuracy: ', accuracy_score(y_test, predictions))
print('precision:', precision_score(y_test, predictions))
print('recall:', recall_score(y_test, predictions))
print('f1-score:', f1_score(y_test, predictions))

##################################KNeighborsClassifier###########################

kne = KNeighborsClassifier()
kne.fit(x_train,y_train)
predict_kne = kne.predict(x_test)
print(classification_report(y_test, predict_kne))
conf_m_lr = confusion_matrix(y_test, predict_kne)
print('KNeighborsClassifier')
print('accuracy: ', accuracy_score(y_test, predict_kne))
print('precision:', precision_score(y_test, predict_kne))
print('recall:', recall_score(y_test, predict_kne))
print('f1-score:', f1_score(y_test, predict_kne))
sns.heatmap(conf_m_lr, annot=True)
plt.title('Матрица KNeighborsClassifier')
plt.xlabel('')
plt.ylabel('')
plt.show()

##################################LinearDiscriminantAnalysis###########################

lda = LinearDiscriminantAnalysis()
lda.fit_transform(x_train,y_train)
predict_lda =lda.predict(x_test)
print(classification_report(y_test, predict_lda))
conf_m_lr = confusion_matrix(y_test, predict_lda)
print('LDA')
print('accuracy: ', accuracy_score(y_test, predict_lda))
print('precision:', precision_score(y_test, predict_lda))
print('recall:', recall_score(y_test, predict_lda))
print('f1-score:', f1_score(y_test, predict_lda))

##################################SVC###########################
sv = svm.SVC()
sv.fit(x_train,y_train)
predict_sv = sv.predict(x_test)
print(classification_report(y_test, predict_sv))
conf_m_lr = confusion_matrix(y_test, predict_sv)
print('SVC')
print('accuracy: ', accuracy_score(y_test, predict_sv))
print('precision:', precision_score(y_test, predict_sv))
print('recall:', recall_score(y_test, predict_sv))
print('f1-score:', f1_score(y_test, predict_sv))
###################################################################################################

##PCA 
sc = StandardScaler() 
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
pca = PCA(n_components = 2) 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test) 
explained_variance = pca.explained_variance_ratio_
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Predicting the training set
# result through scatter plot
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, lm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('PCA')
plt.xlabel('') # for Xlabel
plt.ylabel('') # for Ylabel
plt.legend() # to show legend
# show scatter plot
plt.savefig('C:/TLS-Malware-Detection-with-Machine-Learning-master/diagrams/PCA.png')
plt.show()

# Функция для построения графика наиболее важных изспользуемых признаков для классификации
def plot_feature_importance(importance,names,model_type):

    #Создаваем массивы на основе важности объектов и их названий
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Создаем DataFrame с помощью словаря
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    # Отсортируем DataFrame в порядке убывания важности функции
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    # Определим размер гистограммы
    plt.figure(figsize=(10,8))
    # Построим столбчатую диаграмму Searborn
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.ylim(0, 50)
    # Добавим метки диаграммы
    plt.title(model_type)
    plt.xlabel('ВАЖНОСТЬ ПРИЗНАКОВ')
    plt.ylabel('НАЗВАНИЯ ПРИЗНАКОВ')
    plt.savefig('C:/diagrams/ВАЖНОСТЬ_ПРИЗНАКОВ.png')

plot_feature_importance(rf_clf.feature_importances_,x_train.columns,'СЛУЧАЙНЫЙ ЛЕС')

# Построение графика, группированного по количеству расширений
x1 = reduced_dataset.loc[reduced_dataset.isMalware==1, 'num_of_exts']
x2 = reduced_dataset.loc[reduced_dataset.isMalware==0, 'num_of_exts']

kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(x1, color="blue", label="Зловредный", **kwargs, kde=False)
sns.distplot(x2, color="red", label="Нормальный", **kwargs, kde=False)

plt.legend()
plt.savefig('C:/TLS-Malware-Detection-with-Machine-Learning-master/diagrams/Количество расширений.png')
plt.show()


# Построение графика, группированного по количеству Dst_Port
x1 = reduced_dataset.loc[reduced_dataset.isMalware==1, 'Dst_Port']
x2 = reduced_dataset.loc[reduced_dataset.isMalware==0, 'Dst_Port']

kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(x1, color="blue", label="Зловредный", **kwargs, kde=False)
sns.distplot(x2, color="red", label="Нормальный", **kwargs, kde=False)

plt.legend()
plt.savefig('C:/TLS-Malware-Detection-with-Machine-Learning-master/diagrams/Количество Dst_Port.png')
plt.show()

# Построение графика, группированного по исходным портам
x1 = reduced_dataset.loc[reduced_dataset.isMalware==1, 'Src_Port']
x2 = reduced_dataset.loc[reduced_dataset.isMalware==0, 'Src_Port']

kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(x1, color="blue", label="Зловредный", **kwargs, kde=False)
sns.distplot(x2, color="red", label="Нормальный", **kwargs, kde=False)

plt.legend()
plt.savefig('C:/TLS-Malware-Detection-with-Machine-Learning-master/diagrams/График Портов.png')
plt.show()


# Сохраняем модели (Сохраняем наилучшую по показателем модель, в данном случае модель Случайного леса)
#joblib.dump(rf_clf, 'random_forest_model.pkl')
with open('random_forest.pkl', 'wb') as file:
	pickle.dump(lda, file)

#with open('random_forest_model2.pkl', 'wb') as file:
	pickle.dump(rf_clf, file)
        
#joblib.dump(lm, 'logistics_regression_classifier.pkl')
# with open('logistics_regression_classifier.pkl', 'wb') as file:
# 	pickle.dump(lm, file)

# # Сохраняем модели Случайного леса и Логистической регрессии
joblib.dump(rf_clf, 'random_forest_model.pkl')
joblib.dump(lm, 'logistics_regression_classifier.pkl')

# Загружаем модели из файла
rf_from_joblib = joblib.load('random_forest_model.pkl')
lm_from_joblib = joblib.load('logistics_regression_classifier.pkl')

# Используем загруженную модель, чтобы делать прогнозы
# results = rf_from_joblib.predict(x_test)
# #results = lm_from_joblib.predict(x_test)
# print('Оценка точности классификатора Случайного леса: ', accuracy_score(y_test, results))
