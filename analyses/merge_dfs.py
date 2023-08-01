import pandas as pd




if __name__ == "__main__":

    path = 'analyses/results/'

    df1 = pd.read_csv(path+"benchmark_total.csv")
    df2 = pd.read_csv(path+"results_pwl_cat_classif.csv")
    df3 = pd.read_csv(path+"results_pwl_el_cat_classif.csv")
    df4 = pd.read_csv(path+"results_qe_cat_classif.csv")

    print("All dataframes loaded")

    df2['benchmark'] = 'categorical_classification_medium'
    df3['benchmark'] = 'categorical_classification_medium'
    df4['benchmark'] = 'categorical_classification_medium'

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
    df3['data__keyword'] = df3['data__keyword'].map(data_name_dict)
    df4['data__keyword'] = df4['data__keyword'].map(data_name_dict)

    df2['model_name'] = df2['model_name'].map({'rtdl_mlp_pwl': 'MLP_PieceWiseLinear'})
    df3['model_name'] = df3['model_name'].map({'rtdl_mlp_pwl': 'MLP_PieceWiseLinear_ExtraLayer'})
    df4['model_name'] = df4['model_name'].map({'rtdl_mlp_pwl': 'MLP_QuantizationEmbedding'})

    print("Dataframes mapped")

    # Merge dataframes
    df_merged = pd.concat([
        df1, 
        df2.groupby('data__keyword').head(1000),
        df3.groupby('data__keyword').head(1000),
        df4.groupby('data__keyword').head(1000)
    ], axis=0)

    print("Dataframes merged")

    # Write to csv
    df_merged.to_csv(path+"df_merged.csv", index=False)

    print("Dataframe written to csv")