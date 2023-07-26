import pandas as pd




if __name__ == "__main__":

    path = 'analyses/results/'

    df1 = pd.read_csv(path+"benchmark_total.csv")
    df2 = pd.read_csv(path+"results_pwl_cat_classif.csv")

    df2['benchmark'] = 'categorical_classification_medium'

    data_name_dict = {
        361110: 'electricity',
        361111: 'eye_movements',
        361113: 'covertype',
        361282: 'albert', 
        361283: 'default-of-credit-card-clients', 
        361285: 'road-safety', 
        361286: 'compas-two-years'
    }

    df2['data__keyword'] = df2['data__keyword'].map(data_name_dict)

    new_model_name_dict = {
        'rtdl_mlp_pwl': 'MLP_PieceWiseLinear'
    }

    df2['model_name'] = df2['model_name'].map(new_model_name_dict)

    # Merge dataframes
    df_merged = pd.concat([df1, df2.groupby('data__keyword').head(500)], axis=0)

    # Write to csv
    df_merged.to_csv(path+"df_merged.csv", index=False)