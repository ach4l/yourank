import pandas as pd

def score_fun(weight_list,df_raw):
    orig_total = [20,30,20,30,35,40,15,10,40,15,25,20,30,30,20,20,100]
    weight_list_norm = []
    for i in range(len(weight_list)):
        weight_norm = weight_list[i]/orig_total[i]
        weight_list_norm.append(weight_norm)
    list_of_columns = ["SS","FSR","FQE","FRU","PU","QP","IPR","FPPP","GPH","GUE","GMS","GPHD","RD","WD","ESCS","PCS","PR"]
    df_columns = df_raw[list_of_columns]
    weight = pd.DataFrame(pd.Series(weight_list_norm, index=list_of_columns, name=0))
    df_columns['weighted_sum'] = df_columns.dot(weight)
    df_columns['College'] = df_raw['College']
    df_columns['Original Rank'] = df_raw['Original Rank']
    df_sorted = df_columns.sort_values('weighted_sum',ascending=False)
    return df_sorted