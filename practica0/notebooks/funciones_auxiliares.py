import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dame_variables_categoricas(dataset = None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
        -Descripción: Función que recibe un dataset y devuelve una lista con los nombres de las 
        variables categóricas
        -Inputs: 
            -- dataset: Pandas dataframe que contiene los datos
        -Return:
            -- lista_variables_categoricas: lista con los nombres de las variables categóricas del
            dataset de entrada
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    lista_variables_categoricas = []
    other = []
    
    for i in dataset.columns:
        if (dataset[i].dtype != float) & (dataset[i].dtype != int):
            unicos = int(len(np.unique(dataset[i].dropna(axis = 0, how = "all"))))
            if i in ["fraud_bool", "payment_type", "employment_status", "housing_status", "source", "device_os"]:
                lista_variables_categoricas.append(i)
            else:
                other.append(i)

    return lista_variables_categoricas, other



def plot_feature(df, col_name, isContinuous, target):
    """
    Visualize a variable with and without faceting on the loan status.
    - df dataframe
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True if the variable is continuous, False otherwise
    """
    f, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12,3), dpi = 90)
    
    count_null = df[col_name].isnull().sum()
    
    if isContinuous:
        sns.histplot(df.loc[df[col_name].notnull(), col_name], kde = False, ax = ax1)
        
    else:
        sns.countplot(x = df[col_name], order = sorted(df[col_name].unique()), color = '#5975A4', saturation = 1, ax = ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name + ' Numero de nulos: ' + str(count_null))
    plt.xticks(rotation = 90)


    if isContinuous:
        sns.boxplot(x = col_name, y = target, data = df, ax = ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by ' + target)
    else:
        data = df.groupby(col_name)[target].value_counts(normalize = True).to_frame('proportion').reset_index() 
        data.columns = [col_name, target, 'proportion']
        sns.barplot(x = col_name, y = 'proportion', hue = target, data = data, saturation = 1, ax = ax2)
        ax2.set_ylabel(target + ' fraction')
        ax2.set_title(target)
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()

    
    
def get_deviation_of_mean_perc(df, list_var_continuous, target, multiplier):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    df_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = df[i].mean()
        series_std = df[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = df[i].size
        
        perc_goods = df[i][(df[i] >= left) & (df[i] <= right)].size/size_s
        perc_excess = df[i][(df[i] < left) | (df[i] > right)].size/size_s
        
        if perc_excess > 0:    
            df_concat_percent = pd.DataFrame(df[target][(df[i] < left) | (df[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            df_concat_percent.columns = ["no_fraud", "fraud"]
            df_concat_percent = df_concat_percent.drop(df_concat_percent.index[0])
            df_concat_percent['variable'] = i
            df_concat_percent['sum_outlier_values'] = df[i][(df[i] < left) | (df[i] > right)].size
            df_concat_percent['porcentaje_sum_outlier_values'] = perc_excess
            df_final = pd.concat([df_final, df_concat_percent], axis = 0).reset_index(drop = True)
            
    if df_final.empty:
        print('No existen variables con valores nulos')
        
    return df_final



def get_corr_matrix(dataset = None, metodo = 'pearson', size_figure = [10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0



def get_corr_matrix(dataset = None, metodo = 'pearson', size_figure = [10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0



def get_percent_null_values_target(df, list_var_continuous, target):

    df_final = pd.DataFrame()
    for i in list_var_continuous:
        if i in ["prev_address_months_count", "current_address_months_count", "bank_months_count",
            "session_length_in_minutes", "device_distinct_emails_8w", "intended_balcon_amount"]:
            df_concat_percent = pd.DataFrame(df[target][df[i].isnull()]\
                                            .value_counts(normalize = True).reset_index()).T
            df_concat_percent.columns = ["no_fraud", "fraud"]
            df_concat_percent = df_concat_percent.drop(df_concat_percent.index[0])
            df_concat_percent['variable'] = i
            df_concat_percent['sum_null_values'] = df[i].isnull().sum()
            df_concat_percent['porcentaje_sum_null_values'] = df[i].isnull().sum()/df.shape[0]
            df_final = pd.concat([df_final, df_concat_percent], axis = 0).reset_index(drop = True)
            
    if df_final["sum_null_values"].sum() == 0:
        return print('No existen variables con valores nulos')
        
    return df_final