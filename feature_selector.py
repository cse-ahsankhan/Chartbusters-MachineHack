# Performing Feature Selection
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
def feature_selection(X,y):

    xgb = XGBRegressor()
    estimator = RFE(xgb, 7, step=10, verbose=2)
    selection = estimator.fit(X,y)
    selection.support_
    selection.ranking_
    selected = list(zip(df.columns, selection.support_))
    features = []
    for i in selected:
        if i[1]==True:
            features.append(str(i[0]))
    return(features)
