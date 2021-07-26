from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np

def DfPrepPipeline(df_predict, df_train_Cols, minVec, maxVec):
    # Add new features
    df_predict['BalanceSalaryRatio'] = df_predict.Balance / df_predict.EstimatedSalary
    df_predict['TenureByAge'] = df_predict.Tenure / (df_predict.Age - 18)
    df_predict['CreditScoreGivenAge'] = df_predict.CreditScore / (df_predict.Age - 18)
    # Reorder the columns
    continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                       'BalanceSalaryRatio',
                       'TenureByAge', 'CreditScoreGivenAge']
    cat_vars = ['HasCrCard', 'IsActiveMember', "Geography", "Gender"]
    df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]
    # Change the 0 in categorical variables to -1
    df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
    df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
    # One hot encode the categorical variables
    lst = ["Geography", "Gender"]
    remove = list()
    for i in lst:
        for j in df_predict[i].unique():
            df_predict[i + '_' + j] = np.where(df_predict[i] == j, 1, -1)
        remove.append(i)
    df_predict = df_predict.drop(remove, axis=1)
    # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
    L = list(set(df_train_Cols) - set(df_predict.columns))
    for l in L:
        df_predict[str(l)] = -1
        # MinMax scaling coontinuous variables based on min and max from the train data
    df_predict[continuous_vars] = (df_predict[continuous_vars] - minVec) / (maxVec - minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    df_predict = df_predict[df_train_Cols]
    return df_predict


def best_model(model):
    print(model.best_score_)
    print(model.best_params_)
    print(model.best_estimator_)


def get_auc_scores(y_actual, method, method2):
    auc_score = roc_auc_score(y_actual, method);
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2);
    return (auc_score, fpr_df, tpr_df)
