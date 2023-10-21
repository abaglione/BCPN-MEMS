#!/usr/bin/env python
# coding: utf-8

# # Loading

# In[1]:


# IO
from pathlib import Path

# Utility Libraries
import re
import itertools

# Data Processing
import numpy as np
import pandas as pd

# Feature Engineering
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

# Predictive Analytics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from bcpn_pipeline import data, features, models, consts

# Viz
import seaborn as sns
# sns.set_style("whitegrid")

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
# plt.rcParams.update({'figure.facecolor': [1.0, 1.0, 1.0, 1.0]})



# In[2]:


# Load the data
datafile = Path.joinpath(consts.DATA_PATH, 'final_merged_set_v6.csv')
df = pd.read_csv(datafile, parse_dates=False)
df.head()


# # Data Cleaning & Feature Engineering
# Thank you to Jason Brownlee
# https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/

# In[3]:


# Instantiate a Dataset class
dataset = data.Dataset(df, id_col = 'PtID')
dataset


# In[4]:


# -------- Perform an initial cleaning of the dataset ----------
dataset.clean(to_rename = {**consts.RENAMINGS['demographics'], 
                             **consts.RENAMINGS['medical']}, 
              to_drop=[col for col in dataset.df.columns if '_Name' in col] + 
                      ['MemsNum', 'Monitor', 'pre_dx_date'],
              to_map = consts.CODEBOOK,
              to_binarize = ['race_other'],
              onehots_to_reverse = ['race_']
             )

''' Set dtypes on remaining columns
For now, naively assume we only have numerics, datetimes, or objects
'''
dtypes_dict = {
    'numeric': [col for col in dataset.df.columns if 'date' not in col.lower()],
    'datetime': ['DateEnroll'],
    'categorical': list(set(list(consts.CODEBOOK.keys()) + \
                            ['race', 'education', 'birth_country',
                               'marital_status', 'employment', 'income',
                               'primary_language', 'race',
                               'stage'
                            ]
                           )
                       )
    }

dataset.set_dtypes(dtypes_dict)
dataset.df.head()


# ## Dynamic (Temporal) Features

# Extract temporal features by converting main dataset's df from wide-form to long-form.

# In[5]:


rows = []
# Get a list of all date columns
date_cols = list(dataset.df.filter(regex='date\d{3}$').columns)

i = 0
for col in date_cols:

    # Find all the time cols for that date col
    time_cols = list(dataset.df.filter(
        regex='MEMS_{date_col}_time\d{{1}}$'.format(date_col=col)).columns)  

    ''' Perform a melt so we get MEMS events stratified by patient
        Be sure to include the "within range" column as one of the id_vars''' 
    additional_cols = [
        {
            'original': 'MEMS_' + col + '_numtimes',
            'new': 'num_times_used_today'
        }
    ]
    if i > 0: # The first date won't have an interval or withinrange
        additional_cols.append(
            {
                'original': 'MEMS_' + col + '_interval',
                'new': 'interval'
            }
        )
        additional_cols.append(
            {
                'original': 'MEMS_' + col + '_withinrange',
                'new': 'withinrange'
            }
        )
    
    all_id_col = [dataset.id_col, 'DateEnroll', col] + [x['original'] for x in additional_cols]
    
    res = dataset.df[all_id_col + time_cols].melt(id_vars = all_id_col)
    
    # Tidy up the resulting dataframe
    res.rename(columns={col: 'date', 'value': 'time', 'variable': 'MEMS_day'}, 
               inplace=True)

    res['MEMS_day'] =  res['MEMS_day'].apply(lambda x: int(re.sub(r'_time\d*$', '', x.split('MEMS_date')[1])))
    
    res.rename(columns={x['original']:x['new'] for x in additional_cols},
               inplace=True)

#     res.drop(columns=['variable'], inplace=True)
    
    rows.append(res) # TODO - double check this...getting a weird warning about index alignment
    i += 1

horizons_df = pd.concat(rows, axis=0)

# Create combined datetime column
horizons_df['datetime'] = horizons_df.apply(
    lambda x: features.get_datetime_col(x), axis=1
)
horizons_df['datetime'] = pd.to_datetime(horizons_df['datetime'], errors='coerce')

# Fix dtypes
horizons_df[['withinrange', 'num_times_used_today']] = horizons_df[['withinrange', 'num_times_used_today']].fillna(0).astype(int)
horizons_df['date'] = pd.to_datetime(horizons_df['date'], errors='coerce')
horizons_df['interval'] = pd.to_timedelta(horizons_df['interval']) # Handle NaT intervals for first day?

'''Drop rows with an empty date column.
  Do NOT drop empty time columns - may have dates where it is recorded that the patient
  did not use the cap. So, would have a date but no time. Need this info to calculate
  additional stats later
''' 
horizons_df.dropna(subset=['date'], inplace=True)

# Drop duplicates - these must have been introduced with the melt and with how Kristi's original data was structured
horizons_df.drop_duplicates(inplace=True)

# Remove observations that occurred before a subject's enrollment date
horizons_df = horizons_df.loc[horizons_df['DateEnroll'] < horizons_df['date']]


# ### Quick Look at Adherence For Whole Study

# In[6]:


horizons_df.head()


# In[7]:


len(horizons_df.PtID.unique())
# So, 3 were dropped for insufficient data


# In[8]:


# All remaining participants
adherence = horizons_df.groupby('PtID').agg(
    adherent=(
        'withinrange', 
        lambda x: 1 if x.mean() >= 0.8 else 0
    ),
).reset_index()
adherence.head()


# In[9]:


adherence.shape[0]


# In[10]:


print(f"{round(adherence['adherent'].mean() * 100, 2)}% of participants were adherent across the whole study.")


# In[11]:


navigation_ids = [13,26,32,34,37,38,41,45,52,53,56,64,66,72,76,82]
len(navigation_ids)


# In[12]:


navigation = adherence[adherence['PtID'].isin(navigation_ids)]
print(f"{round(navigation['adherent'].mean() * 100, 2)}% of participants who received patient navigation were adherent across the whole study.")


# In[13]:


no_navigation = adherence[~adherence['PtID'].isin(navigation_ids)]
print(f"{round(no_navigation['adherent'].mean() * 100, 2)}% of participants who did not receive patient navigation were adherent across the whole study.")


# ### Generate new features

# In[14]:


# Add binary indicator of any usage (not just number of times used) on a given day
horizons_df['used_today'] = horizons_df['num_times_used_today'].apply(
    lambda x: 1 if x > 0 else 0
)

'''Generate horizons of interest (time of day, weekday, day/month of study, etc)
   'time_of_day' category gets automatically encoded as a Categorical
''' 
horizons_df = features.get_temporal_feats(df=horizons_df, start_date_col='DateEnroll', 
                                          id_col='PtID', time_of_day_props=consts.TIME_OF_DAY_PROPS)

# washout period of 1 month, per Kristi's recommendation
horizons_df = horizons_df[horizons_df['study_month'] > 0].reset_index(drop=True)
horizons_df['study_month'].min() # verify


# In[15]:


''' Quick fix for duplicates that are introduced...
      not sure of a better way to do this yet'''
df = horizons_df[horizons_df.duplicated(['PtID', 'study_day', 'interval'])]
df = df[df['num_times_used_today'] <= 1]
horizons_df.drop(df.index, axis=0, inplace=True)
horizons_df


# In[16]:


horizons_df


# In[17]:


def get_col_mode(x):
    m = x.mode()
    
    if isinstance(m, str):
        return m
    else:
        try:
            first_mode = m[0]
            return first_mode
        except Exception as e:
            return np.nan

temporal_featuresets = list()
'''Group by our desired horizon and add standard metrics such as mean, std
'''

for horizon in consts.TARGET_HORIZONS:
    nominal_cols = []
    groupby_cols = [dataset.id_col, horizon]
    
    # Get the total number of events for the given horizon
    df = horizons_df.groupby(groupby_cols).agg(
        n_events=('num_times_used_today', 'sum')
    ).reset_index()
    
    
    if horizon == 'study_day':
        cols = ['is_weekday', 'day_of_week']
        df2 = horizons_df[groupby_cols + cols]
        df = df.merge(df2, on=groupby_cols, how='inner')
        
        # Add columns indicating if the MEMS cap was used during a given time(s) of day
        # Basically a manual one-hot encoding while we're here
        col = 'time_of_day'
        df2 = horizons_df.groupby(groupby_cols)[col].value_counts().reset_index(name='count')
        df2.rename(columns={'level_2': col}, inplace=True)
        
        df2 = df2.pivot_table(
            columns=col, index=['PtID', 'study_day'], values='count'
        ).reset_index().rename_axis(None, axis=1)
        
        df = df.merge(df2, on=groupby_cols, how='inner')
        for col in consts.TIME_OF_DAY_PROPS['labels']:
            df[col] = pd.to_numeric(df[col].fillna(0).apply(lambda x: 1 if x > 0 else x))
        
        cols = {x: 'time_of_day_' + x for x in consts.TIME_OF_DAY_PROPS['labels']}
        df.rename(columns=cols,
                  inplace=True)
        
        # Add to list of nominal cols after one-hot encoding
        nominal_cols += [col for col in cols.values()]
        
    else:
        if horizon == 'study_week':
            denom = consts.DAYS_IN_WEEK
        else:
            denom = consts.DAYS_IN_MONTH
        
        # Get standard temporal metrics
        df2 = features.calc_standard_temporal_metrics(horizons_df, groupby_cols, 'datetime')
        df = df.merge(df2, on=groupby_cols, how='inner')

        # Calculate avg and standard deviation of number of times used
        df2 = horizons_df.groupby(groupby_cols + ['study_day'])['num_times_used_today'].max().reset_index()
        df2 = horizons_df.groupby(groupby_cols).agg(
            num_daily_events_mean=('num_times_used_today', lambda x: x.sum() / denom)
        ).reset_index()

        df = df.merge(df2, on=groupby_cols, how='inner')

        # Get most common time of day of event occurence
        df2 = horizons_df.groupby(groupby_cols).agg(
            event_time_of_day_mode=('time_of_day', get_col_mode)
        ).reset_index()

        df = df.merge(df2, on=groupby_cols, how='inner')
        
        # Explicitly set dtype so we can later select and one-hot encode
        df['event_time_of_day_mode'] = df['event_time_of_day_mode'].astype('category') 

    # Calculate adherence rate
    if 'day' in horizon:
        df2 = horizons_df.groupby(groupby_cols).agg(
            adherence_rate=('withinrange', 'max')
        ).reset_index()
    else:
        df2 = horizons_df.groupby(groupby_cols + ['study_day'])['withinrange'].max().reset_index() # Max will be 1 or 0
        df2 = df2.groupby(groupby_cols).agg(
            adherence_rate=('withinrange', lambda x: x.sum() / denom)
        ).reset_index()
    
    df = df.merge(df2, on=groupby_cols, how='inner')
    
    # Help pandas since it doesn't process datetimes well and introduces duplicate entries on merges
    df = df.drop_duplicates(subset=groupby_cols)
    
    # Create a featureset from the resulting dataframe
    temporal_featuresets.append(features.Featureset(df=df,
                                                    name=horizon, #Intentional for now - using horizon as name
                                                    id_col=dataset.id_col,
                                                    horizon=horizon,
                                                    nominal_cols = nominal_cols))


# In[18]:


# Sanity check
temporal_featuresets[0].df.select_dtypes('category')


# In[19]:


temporal_featuresets[0].df.iloc[-1]


# In[20]:


temporal_featuresets[1].nominal_cols


# ### Examine Collinearity and Variance Inflation Factor

# In[21]:


# Study Day
df = temporal_featuresets[0].df
df.select_dtypes('number').corr()


# In[22]:


df2 = df.select_dtypes('number').drop(columns=['PtID', 'adherence_rate'])
vif_data = pd.DataFrame()
vif_data["feature"] = df2.columns
vif_data["VIF"]  = [vif(df2.values, i) for i in range(len(df2.columns))]
vif_data


# In[23]:


# Drop a few to see if we improve
df2 = df.select_dtypes('number').drop(columns=['PtID', 'adherence_rate', 'n_events'])
vif_data = pd.DataFrame()
vif_data["feature"] = df2.columns
vif_data["VIF"]  = [vif(df2.values, i) for i in range(len(df2.columns))]
vif_data

# Yep - looks much better


# In[24]:


# Drop in the actual featureset and verify it's been dropped
temporal_featuresets[0].df.drop(columns=['n_events'], inplace=True)
print(temporal_featuresets[0].df.columns)
temporal_featuresets[0]


# In[25]:


# Study Week
df = temporal_featuresets[1].df.select_dtypes('number')
df.corr()


# In[26]:


df2 = df.select_dtypes('number').drop(columns=['PtID', 'adherence_rate'])
vif_data = pd.DataFrame()
vif_data["feature"] = df2.columns
vif_data["VIF"]  = [vif(df2.values, i) for i in range(len(df2.columns))]
vif_data


# In[27]:


# Drop n_events, n_daily_events mean, and between event time mean to see if we improve
df2 = df.select_dtypes('number').drop(columns=['n_events', 'between_event_time_mean', 'num_daily_events_mean', 'PtID', 'adherence_rate'])
vif_data = pd.DataFrame()
vif_data["feature"] = df2.columns
vif_data["VIF"]  = [vif(df2.values, i) for i in range(len(df2.columns))]
vif_data

# Yep - looks much better


# In[28]:


# Drop in the actual featureset and verify it's been dropped
temporal_featuresets[1].df.drop(columns=['n_events', 'between_event_time_mean', 'num_daily_events_mean'], inplace=True)
print(temporal_featuresets[1].df.columns)
temporal_featuresets[1]


# In[29]:


for fs in temporal_featuresets:
    print(fs.name)
    for col in fs.df.columns:
        if col != fs.id_col:
            try:
                plt.hist(fs.df[col])
                plt.title(col)
                plt.show()
            except:
                pass


# # Prediction

# In[30]:


# Sanity check - this column should NOT be in the final set
# static_featuresets[0].df['total_days_8']


# In[31]:


target_col = 'adherent'

'''Set the target columns on the temporal featuresets
   This step should be run before doing any experiments
''' 
for t_feats in temporal_featuresets:
    
    #Convert adherence_rate into a binary indicator of adherence
    t_feats.df[target_col] = t_feats.df['adherence_rate'].apply(
        lambda x: 1 if x > consts.ADHERENCE_THRESHOLD else 0
    )
    # Drop the original column
    t_feats.df.drop(columns=['adherence_rate'], inplace=True)

    # Set the target col
    t_feats.target_col = target_col
    
    '''For now, treat the target as a nominal column
      This is because, if we use lagged values of the target (e.g., adherent),
         those columns will be considered nominal, and we'll need to use the target col as a name
         against which to find and check those columns.
      There's a safeguard in place later, in the get_lagged_featureset function, 
        to ensure the target itself is NOT included in the final list of nominal columns.'''
    t_feats.nominal_cols += [target_col]


# In[36]:


temporal_featuresets[0].df


# In[53]:



# ## Study: Predict Adherence from MEMS Data Only

# ### Tune number of lags

# In[ ]:


''' Test the model performance for a range of lags (number of previous inputs)
      and range of max_depths (since training with RF by default)
    max_depth exploration will help ensure we aren't overfitting.
'''
# for t_feats in temporal_featuresets:
#     models.tune_lags(t_feats)


# In[64]:


# Plot and analyze results

# results = []

# for f in consts.OUTPUT_PATH_LAGS.glob('*final_clf_pred.csv'):
#     df = pd.read_csv(f)
#     results.append(df)

# results = pd.concat(results, axis=0).reset_index(drop=True)
# results.drop(columns=['Unnamed: 0'], inplace=True)
# results.rename(columns={'featureset': 'Feature Set'}, inplace=True)
# results


# # In[65]:


# for col in ['n_lags', 'accuracy', 'max_depth']:
#     results[col] = pd.to_numeric(results[col])
# results


# # In[66]:


# plt.rcParams.update({'font.size': 14})


# # In[67]:


# # '''
# # Takeaways:

# # - study_day: 3 lags with max_depth 2 best for study_day - some overfitting, but not dramatic. 
# # Overfitting worsens beyond this.

# # - study_week: 3 lags with shallow tree (max_depth=2) best for study week. 
# # Overfitting worsens after that, with no major gain in performance
# # '''

# fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
# results.rename(columns={col: col.capitalize() for col in ['type']}, inplace=True)
# for max_depth in range(1, 3):
#     df = results[results['max_depth'] == max_depth]
    
#     g = sns.lineplot(x='n_lags',y='specificity', hue='Feature Set', 
#                      style='Type', data=df, ax=axes[max_depth-1],
#                      legend=(max_depth==2))
#     g.set( ylim=(0, 1), title='Max Depth: ' + str(max_depth), ylabel='Specificity', xlabel='Lags')
    
# axes[0].set(xlabel='(A)')
# axes[1].legend(loc=(1.1, 0.5))
# axes[1].set(xlabel='(B)')
# plt.savefig(Path.joinpath(consts.OUTPUT_PATH_LAGS, 'spec_md1_2.png'), bbox_inches='tight',
#            facecolor='white', transparent=True
#            )
# plt.show()

# fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)


# for max_depth in range(3, 6):
#     df = results[results['max_depth'] == max_depth]
    
#     g = sns.lineplot(x='n_lags',y='specificity', hue='Feature Set', 
#                      style='Type', data=df, ax=axes[max_depth-3],
#                      legend=(max_depth==5))
#     g.set( ylim=(0, 1), title='Max Depth: ' + str(max_depth), ylabel='Specificity', xlabel='Lags')
    
# axes[2].legend(loc=(1.1, 0.5))
# plt.savefig(Path.joinpath(consts.OUTPUT_PATH_LAGS, 'spec_md3_5.png'), bbox_inches='tight',
#            facecolor='white', transparent=True
#            )
# plt.show()


# ### Do prediction task

# In[ ]:


# # ----- Now predict using optimal number of lags for each horizon--- 
for t_feats in temporal_featuresets:    
    
    ''' These max_depth are for the default (baseline) classifiers only.
    Useful for benchmarking studies.'''

    fs_lagged = t_feats.prep_for_modeling(n_lags=3)
    models.predict_from_mems(fs_lagged, max_depth=2)       

