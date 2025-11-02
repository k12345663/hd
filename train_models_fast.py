import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

df = pd.read_csv('data/heart.csv')

X = df.drop('target', axis=1)
y = df['target']

cat_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

models = []
models.append(('logreg_liblinear', LogisticRegression(max_iter=1000, solver='liblinear')))
models.append(('logreg_saga', LogisticRegression(max_iter=1000, solver='saga')))
models.append(('ridgecv', RidgeClassifierCV()))
models.append(('sgd_log', SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)))
models.append(('dt_gini', DecisionTreeClassifier(criterion='gini', random_state=0)))
models.append(('dt_entropy', DecisionTreeClassifier(criterion='entropy', random_state=0)))
models.append(('extra_tree', ExtraTreeClassifier(random_state=0)))
models.append(('rf', RandomForestClassifier(n_estimators=100, random_state=0)))
models.append(('et', ExtraTreesClassifier(n_estimators=100, random_state=0)))
models.append(('gb', GradientBoostingClassifier(n_estimators=100, random_state=0)))
models.append(('ada', AdaBoostClassifier(n_estimators=50, random_state=0)))
models.append(('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric='logloss', random_state=0, verbosity=0)))
models.append(('cat', CatBoostClassifier(verbose=0, depth=4, learning_rate=0.1, iterations=100, random_seed=0)))
models.append(('bag_logreg', BaggingClassifier(LogisticRegression(max_iter=1000, solver='liblinear'), n_estimators=10, random_state=0)))
models.append(('bag_tree', BaggingClassifier(DecisionTreeClassifier(random_state=0), n_estimators=20, random_state=0)))
models.append(('histgb', HistGradientBoostingClassifier(max_iter=100, random_state=0)))
models.append(('knn3', KNeighborsClassifier(n_neighbors=3)))
models.append(('knn7', KNeighborsClassifier(n_neighbors=7, weights='distance')))
models.append(('svc_rbf', SVC(kernel='rbf', probability=True, random_state=0)))
models.append(('linear_svc', CalibratedClassifierCV(LinearSVC(random_state=0, max_iter=2000), method='sigmoid', cv=3)))
models.append(('gauss_nb', GaussianNB()))
models.append(('mlp_shallow', MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=0)))
models.append(('mlp_deep', MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=1000, random_state=0)))
models.append(('rf_cal', CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=0), method='isotonic', cv=3)))
models.append(('brf', BalancedRandomForestClassifier(n_estimators=100, random_state=0)))
models.append(('easy_ens', EasyEnsembleClassifier(n_estimators=10, random_state=0)))

base_estimators = [
    ('logreg', LogisticRegression(max_iter=1000, solver='liblinear')),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=0)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=0))
]
stack = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(max_iter=1000), cv=3)
models.append(('stack', stack))

vote_estimators = [
    ('logreg', LogisticRegression(max_iter=1000, solver='liblinear')),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=0)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=0))
]
vote = VotingClassifier(estimators=vote_estimators, voting='soft')
models.append(('vote_soft', vote))

metrics_rows = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

total_models = len(models)
print(f"Training {total_models} models...")

for i, (name, clf) in enumerate(models):
    print(f"Training model {i+1}/{total_models}: {name}")
    pipe = Pipeline(steps=[('prep', preprocess), ('model', clf)])
    
    try:
        y_proba = cross_val_predict(pipe, X, y, cv=skf, method='predict_proba')
        y_pred = (y_proba[:,1] >= 0.5).astype(int)
    except AttributeError:
        y_pred_raw = cross_val_predict(pipe, X, y, cv=skf, method='decision_function')
        y_proba = 1/(1+np.exp(-y_pred_raw))
        if y_proba.ndim == 1:
            y_proba = np.vstack([1-y_proba, y_proba]).T
        y_pred = (y_proba[:,1] >= 0.5).astype(int)
    
    auc = roc_auc_score(y, y_proba[:,1])
    acc = accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred)
    prec = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    brier = brier_score_loss(y, y_proba[:,1])
    
    sample_input = X.iloc[[0]]
    pipe.fit(X, y)
    start = time.time()
    try:
        _ = pipe.predict_proba(sample_input)
    except AttributeError:
        _ = pipe.predict(sample_input)
    latency = (time.time() - start) * 1000
    
    metrics_rows.append({
        'model': name,
        'roc_auc': auc,
        'accuracy': acc,
        'recall': rec,
        'precision': prec,
        'f1': f1,
        'brier': brier,
        'latency_ms': latency
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df.sort_values(by=['recall','roc_auc'], ascending=[False,False]).reset_index(drop=True)

best_model_name = metrics_df.iloc[0]['model']
best_clf = dict(models)[best_model_name]
best_pipe = Pipeline(steps=[('prep', preprocess), ('model', best_clf)])
best_pipe.fit(X, y)

joblib.dump(best_pipe, 'data/best_model.pkl')
metrics_df.to_csv('data/leaderboard_metrics.csv', index=False)
joblib.dump(preprocess, 'data/pipeline.pkl')

print("\n=== Training Complete ===")
print(f"Best model: {best_model_name}")
print(f"Recall: {metrics_df.iloc[0]['recall']:.4f}")
print(f"AUROC: {metrics_df.iloc[0]['roc_auc']:.4f}")
print("\nTop 5 models:")
print(metrics_df.head()[['model','recall','roc_auc','accuracy']])
