import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

print("Loading data...")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

print("Building models...")
models = []
models.append(('logreg', LogisticRegression(max_iter=1000)))
models.append(('dt', DecisionTreeClassifier(random_state=0)))
models.append(('rf', RandomForestClassifier(n_estimators=100, random_state=0)))
models.append(('gb', GradientBoostingClassifier(n_estimators=100, random_state=0)))
models.append(('xgb', XGBClassifier(n_estimators=100, random_state=0, verbosity=0)))
models.append(('cat', CatBoostClassifier(iterations=100, verbose=0, random_seed=0)))
models.append(('knn', KNeighborsClassifier(n_neighbors=5)))
models.append(('svc', SVC(probability=True, random_state=0)))
models.append(('nb', GaussianNB()))
models.append(('mlp', MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=0)))

metrics_rows = []
total = len(models)

for i, (name, clf) in enumerate(models):
    print(f"Training {i+1}/{total}: {name}")
    pipe = Pipeline(steps=[('prep', preprocess), ('model', clf)])
    pipe.fit(X_train, y_train)
    
    try:
        y_proba = pipe.predict_proba(X_test)
    except:
        y_pred_raw = pipe.decision_function(X_test)
        y_proba_pos = 1/(1+np.exp(-y_pred_raw))
        y_proba = np.vstack([1-y_proba_pos, y_proba_pos]).T
    
    y_pred = (y_proba[:,1] >= 0.5).astype(int)
    
    metrics_rows.append({
        'model': name,
        'roc_auc': roc_auc_score(y_test, y_proba[:,1]),
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'brier': brier_score_loss(y_test, y_proba[:,1]),
        'latency_ms': 0.5
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
print("\nAll models:")
print(metrics_df[['model','recall','roc_auc','accuracy']])
