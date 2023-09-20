import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def df_survival_prob(dic_input, model, encoder):
    
    df_input = pd.DataFrame(dic_input, index=['patient_input'])
    list_column_categorical = df_input.select_dtypes(object).columns
    df_input[list_column_categorical] = df_input[list_column_categorical].astype('category')
    df_input = encoder.transform(df_input)

    model_survival = model.predict_survival_function(df_input)[0]

    df_survival = pd.DataFrame({
        'time': model_survival.x,
        'survival_prob': model_survival.y
    })
    
    return df_survival

def survival_prob_median(df_survival):

    idx = df_survival.survival_prob.lt(0.5).idxmax()
    time, survival_prob = df_survival.loc[idx]
    survival_prob = round(survival_prob*100, 2)
    time = round(time, 2)

    return survival_prob, time


def predict_survival_probability(dic_input, model, encoder):
    
    df_survival = df_survival_prob(dic_input, model, encoder)
    survival_prob, time = survival_prob_median(df_survival)
    
    return survival_prob, time