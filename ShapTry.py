#%%
import xgboost
import shap
from sklearn.model_selection import train_test_split as tv_split
from sklearn.metrics import mean_squared_error as mse
# load JS visualization code to notebook
shap.initjs()
#%%
# train XGBoost model
X,y = shap.datasets.boston()
X_train, X_test, y_train, y_test = tv_split(X, y, test_size=0.33)
#%%
model = xgboost.train({"learning_rate": 0.355}, xgboost.DMatrix(X_train, label=y_train), 100)
mse(y_test, model.predict(xgboost.DMatrix(X_test)))
#%%
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

#%%
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

# %%
shap.force_plot(explainer.expected_value, shap_values, X)

# %%
shap.dependence_plot("RM", shap_values, X)

# %%
shap.summary_plot(shap_values, X)

# %%
def pop_fun(X_train, X_test, name):
    Xpop = X_train.copy()
    Xpoptest = X_test.copy()
    Xpop.pop(name)
    Xpoptest.pop(name)
    return Xpop, Xpoptest
# %%
X_pop, X_test_pop = pop_fun(X_train, X_test, "TAX")
model = xgboost.train({"learning_rate": 0.355}, xgboost.DMatrix(X_pop, label=y_train), 100)
mse(y_test, model.predict(xgboost.DMatrix(X_test_pop)))
# %%
