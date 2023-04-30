"""
include:
convert_woe_cat(feature, target='Label')
convert_woe(feature, bins=10, target='Label')
convert_woe_sliced(feature, sliced_location, bins=10, target='Label')
"""
import numpy as np
import pandas as pd

# Weight of evidence: categorical
def convert_woe_cat(feature, df, target='Label'):
   
    # woe calculation
    df_woe = df.groupby(feature).count()[[target]]
    df_woe.columns=['Total']
    df_woe['Good'] = df.groupby(feature).sum(numeric_only =True)[target]
    df_woe['Bad'] = df_woe.Total - df_woe.Good

    good = df_woe.Good
    bad  = df_woe.Bad
    total_good = df_woe.Good.sum()
    total_bad  = df_woe.Bad.sum()
    df_woe['WOE'] = np.log(good/total_good + 0.001) - np.log(bad/total_bad + 0.001)
    df_woe['IV'] = df_woe.WOE*(good/total_good - bad/total_bad)
    print(f'Information value: {df_woe.IV.sum():.2f}')
    return df_woe.sort_values('WOE', ascending=False)

# Weight of evidence
def convert_woe(feature, df, bins=10, target='Label'):
    # set label length if bins input is list (in case of coarse finetuning)
    label_length=0
    if isinstance(bins,list):
        label_length = len(bins)
    else:
        label_length = bins+1
    # bins, range and target count    
    df_bins = pd.DataFrame({
        feature+'_bins': pd.cut(df[feature], bins, labels=np.arange(1, label_length,1)).astype(int).replace(np.inf, 0),
        feature+'_range': pd.cut(df[feature], bins),
        target : df[target]
    })
    # woe calculation
    df_woe = df_bins.groupby(feature+'_range').agg(pd.Series.mode)[[feature+'_bins']]
    df_woe.columns=['Bin']
    df_woe['Total'] = df_bins.groupby(feature+'_range').count()[[target]]
    df_woe['Good'] = df_bins.groupby(feature+'_range').sum(numeric_only =True)[target]
    df_woe['Bad'] = df_woe.Total - df_woe.Good

    good = df_woe.Good
    bad  = df_woe.Bad
    total_good = df_woe.Good.sum()
    total_bad  = df_woe.Bad.sum()
    df_woe['WOE'] = np.log(good/total_good + 0.001) - np.log(bad/total_bad + 0.001)
    df_woe['IV'] = df_woe.WOE*(good/total_good - bad/total_bad)
    print(f'Information value: {df_woe.IV.sum():.2f}')
    return df_woe.sort_values('WOE', ascending=False)

# Weight of evidence (sliced)
def convert_woe_sliced(feature, df, sliced_location, bins=10, target='Label'):
    # set label length if bins input is list (in case of coarse finetuning)
    label_length=0
    if isinstance(bins,list):
        label_length = len(bins)
    else:
        label_length = bins+1
        
    df_bins = pd.DataFrame({
        feature+'_bins': pd.cut(df[feature][sliced_location], bins, labels=np.arange(1, label_length,1)).astype(int),
        feature+'_range': pd.cut(df[feature][sliced_location], bins),
        target : df[target][sliced_location]
    })
    df_woe = df_bins.groupby(feature+'_range').agg(pd.Series.mode)[[feature+'_bins']]
    df_woe.columns=['Bin']
    df_woe['Total'] = df_bins.groupby(feature+'_range').count()[[target]]
    df_woe['Good'] = df_bins.groupby(feature+'_range').sum(numeric_only =True)[target]
    df_woe['Bad'] = df_woe.Total - df_woe.Good

    good = df_woe.Good
    bad  = df_woe.Bad
    total_good = df_woe.Good.sum()
    total_bad  = df_woe.Bad.sum()
    df_woe['WOE'] = np.log(good/total_good + 0.001) - np.log(bad/total_bad + 0.001)
    df_woe['IV'] = df_woe.WOE*(good/total_good - bad/total_bad)
    print(f'Information value: {df_woe.IV.sum():.2f}')
    return df_woe.sort_values('WOE', ascending=False)