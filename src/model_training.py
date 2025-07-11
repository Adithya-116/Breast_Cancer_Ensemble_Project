from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def get_models():
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    voting = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)], voting='soft')
    stacking = StackingClassifier(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)], final_estimator=LogisticRegression(), cv=5)

    return lr, rf, xgb, voting, stacking

def train_models(models, X_train, y_train, X_test, y_test):
    lr, rf, xgb, voting, stacking = models

    voting.fit(X_train, y_train)
    voting_pred = voting.predict(X_test)
    voting_acc = accuracy_score(y_test, voting_pred)

    stacking.fit(X_train, y_train)
    stacking_pred = stacking.predict(X_test)
    stacking_acc = accuracy_score(y_test, stacking_pred)

    if stacking_acc > voting_acc:
        return "StackingClassifier", stacking, stacking_pred, stacking_acc, stacking.predict_proba(X_test)[:, 1]
    else:
        return "VotingClassifier", voting, voting_pred, voting_acc, voting.predict_proba(X_test)[:, 1]