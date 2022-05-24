#!/usr/bin/env python
# coding: utf-8

# # Loading

# In[ ]:


# IO
from pathlib import Path

# Utility Libraries
import re
import itertools

# Data Processing
import pandas as pd

# Predictive Analytics
from bcpn_pipeline import data, features, models, consts

# Viz
import seaborn as sns
sns.set_style("whitegrid")

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
# plt.rcParams.update({'figure.facecolor': [1.0, 1.0, 1.0, 1.0]})



# In[ ]:


# Load the data
datafile = Path("data/final_merged_set_v6.csv")
df = pd.read_csv(datafile, parse_dates=False)
df.head()


# # Data Cleaning & Feature Engineering
# Thank you to Jason Brownlee
# https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/

# In[ ]:


# Instantiate a Dataset class
dataset = data.Dataset(df, id_col = 'PtID')
dataset


# In[ ]:


# -------- Perform an initial cleaning of the dataset ----------
dataset.clean(to_rename = {**consts.RENAMINGS['demographics'], 
                             **consts.RENAMINGS['medical']}, 
              to_drop=[col for col in dataset.df.columns if '_Name' in col] + 
                      ['MemsNum', 'Monitor'],
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
    'categorical': list(consts.CODEBOOK.keys()) + ['race']
}

dataset.set_dtypes(dtypes_dict)
dataset.df.head()


# ## Static Features

# Generate static (non-temporal) features from measures such as validate instruments (e.g., FACTB)

# ### Organize candidate features

# In[ ]:


''' 
 Organize the candidate features into useful categories for later reference
 A bit tedious, but helpful 
'''
feat_categories = {
    'demographics': [v for v in consts.RENAMINGS['demographics'].values()
                     if v in dataset.df.columns] + ['race'], #add the new, single race col
    'medical': [v for v in consts.RENAMINGS['medical'].values() if v in dataset.df.columns] + \
               [col for col in ['early_late', 'diagtoenroll'] 
                if col in dataset.df.columns],
    'scores': []
}
feat_categories


# In[ ]:


''' This dataset has several repeated measures for validated instruments, 
such as the FACTB

Columns for repeated measures for the same instrument share a suffix (e.g., '_FACTB')
Use regex to populate the `scores` category subdictionary quickly, using these suffixes
''' 

for k,v in consts.SCORES.items():
    
    #  Handle special case of BCPT before doing anything else
    if k == 'BCPT':
        dataset.df.drop(
            list(dataset.df.filter(regex = '_BCPT\d*YN$')), 
            axis = 1, 
            inplace = True
        )
        dataset.df.drop(
            list(dataset.df.filter(regex = '_BCPT\d*O$')), 
            axis = 1, 
            inplace = True
        )
    for prefix in consts.SCORE_PREFIXES:
        # Some measures weren't precalculated. Let's fix this.  
        if v['precalculated'] == False:

            #  Get the aggregate score and add it to the dataset as a new column

                score_cols = list(
                    dataset.df.filter(regex='^' + prefix + v['suffix'] + '\d*').columns
                )

                dataset.df[prefix + v['suffix']] = dataset.df[score_cols].sum(axis=1)

        #  We'll include this new column as a feature
        feat_categories['scores'] += [prefix + v['suffix']]


# In[ ]:


''' Create a catch-all category of remaining features, to ensure we got everything '''
other_feats = [col for col in dataset.df.columns 
              if col not in list(itertools.chain(*feat_categories.values())) # exclude anything already in the list
              and not any(prefix in col for prefix in ['A_', 'B_', 'C_']) # exclude individual score cols
              and 'date' not in col 
              and col not in dataset.id_col
             ]

other_feats


# ### Generate new features

# In[ ]:


''' Create new columns for demographic and medical variables
Be sure we update the feature categories dictionary '''

demog_drug_cols = [col for col in dataset.df.columns if 'A_DEMO13DRUG' in col]
newcol = 'DEMOG_numdrugs'
dataset.df[newcol] = dataset.df[demog_drug_cols].count(axis=1)
feat_categories['demographics'] += [newcol]

# Removed post-exam cols - DUH


# ### Create featuresets

# In[ ]:


static_featuresets = list()

'''Create two distinct featuresets - one with demographics + med record data,
   and one with scores '''
fs_combos = [['demographics', 'medical'], ['scores']]

for combo in fs_combos:
    feat_cats_subset = {k:v for k,v in feat_categories.items() if k in combo}
    df = data.build_df_from_feature_categories(dataset.df, feat_cats_subset, dataset.id_col)
    
    ''' We have scores at baseline (0 months), mid-study (4 months) and post-study (8 months).
        Here, we are going to reshape the data so dataframe is long-form, with three observations per subject
        (at 0 months, 4 months, and 8 months, respectively).
        
        We are also going to create a column that indicates the time period(s) for which we want to
        predict adherence. Per Navreet's documentation, surveys at the 4- and 8-month time points 
        asked about experiences in the PAST. So, for the 0-month observations, we will predict 
        some immediate future adherence value(s).But for the 4- and 8-month observations, we will 
        predict some immediate PAST adherence value(s).
    '''
    if 'scores' in combo:
        dfs = []
        
        for prefix in consts.SCORE_PREFIXES:
            i = consts.SCORE_PREFIXES.index(prefix)
            df2= df[['PtID'] + [col for col in df.columns if prefix in col]]
            df2.columns = df2.columns.str.replace(prefix, '')
        
            if i == 0:
                df2['study_day'] = 0
                df2['study_month'] = 0
            elif i == 1: # Prefix B_, the 4-month surveys; predict into the past
                df2['study_day'] = 119
                df2['study_month'] = 3
            else: # Prefix C_, the 8-month surveys; predict into the past
                df2['study_day'] = 239
                df2['study_month'] = 7
            dfs.append(df2)

        df = pd.concat(dfs, axis=0, ignore_index=True)
        
        ''' Remove participants who didn't complete all three questionnaires 
        Thank you Wes McKinney
        https://stackoverflow.com/questions/14016247/find-integer-index-of-rows-with-nan-in-pandas-dataframe
        '''
        idx = pd.isnull(df).any(1).to_numpy().nonzero()[0]
        ptids_to_exclude = list(df.iloc[idx, :]['PtID'].unique())
        df = df.loc[~df['PtID'].isin(ptids_to_exclude)]
    
    else:
        for horizon in consts.TARGET_HORIZONS:
            df[horizon] = 0
    static_featuresets.append(features.Featureset(df=df, name=' + '.join(combo), id_col = dataset.id_col))
        
static_featuresets


# ## Dynamic (Temporal) Features

# Extract temporal features by converting main dataset's df from wide-form to long-form.

# In[ ]:


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


# ### Generate new features

# In[ ]:


# Add binary indicator of any usage (not just number of times used) on a given day
horizons_df['used_today'] = horizons_df['num_times_used_today'].apply(
    lambda x: 1 if x > 0 else 0
)

'''Generate horizons of interest (time of day, weekday, day/month of study, etc)
   'time_of_day' category gets automatically encoded as a Categorical
''' 
horizons_df = features.get_temporal_feats(df=horizons_df, start_date_col='DateEnroll', 
                                          id_col='PtID', time_of_day_props=consts.TIME_OF_DAY_PROPS)
horizons_df


# In[ ]:


''' Quick fix for duplicates that are introduced...
      not sure of a better way to do this yet'''
df = horizons_df[horizons_df.duplicated(['PtID', 'study_day', 'interval'])]
df = df[df['num_times_used_today'] <= 1]
horizons_df.drop(df.index, axis=0, inplace=True)
horizons_df


# In[ ]:


temporal_featuresets = list()
'''Group by our desired horizon and add standard metrics such as mean, std
'''

for horizon in consts.TARGET_HORIZONS:
    nominal_cols = []
    groupby_cols = [dataset.id_col, horizon]
    
    df = horizons_df.groupby(groupby_cols)['datetime'].agg({
        'n_events': 'count'
    }).reset_index()
    
    if horizon == 'study_day':
        cols = ['is_weekday', 'day_of_week']
        df2 = horizons_df[groupby_cols + cols]
        df = df.merge(df2, on=groupby_cols, how='outer')
        nominal_cols += cols
        
        # Add columns indicating if the MEMS cap was used during a given time(s) of day
        # Basically a manual one-hot encoding while we're here
        col = 'time_of_day'
        df2 = horizons_df.groupby(groupby_cols)[col].value_counts().reset_index(name='count')
        df2 = df2.reset_index().pivot_table(
            columns=col, index=['PtID', 'study_day'], values='count'
        ).reset_index().rename_axis(None, axis=1)
        
        df = df.merge(df2, on=groupby_cols, how='outer')
        
        for col in consts.TIME_OF_DAY_PROPS['labels']:
            df[col] = pd.to_numeric(df[col].fillna(0).apply(lambda x: 1 if x > 0 else x))
            nominal_cols += [col]
            
        df.rename(columns={x: 'time_of_day_' + x for x in consts.TIME_OF_DAY_PROPS['labels']},
                  inplace=True)
    else:
        if horizon == 'study_week':
            denom = consts.DAYS_IN_WEEK
        else:
            denom = consts.DAYS_IN_MONTH
        
        # Get standard temporal metrics
        df2 = features.calc_standard_temporal_metrics(horizons_df, groupby_cols, 'datetime')
        df = df.merge(df2, on=groupby_cols, how='outer')

        # Calculate avg and standard deviation of number of times used
        df2 = horizons_df.groupby(groupby_cols + ['study_day'])['num_times_used_today'].max().reset_index()
        df2 = horizons_df.groupby(groupby_cols)['num_times_used_today'].agg({
            'num_daily_events_mean': lambda x: x.sum() / denom
        }).reset_index()
        df = df.merge(df2, on=groupby_cols, how='outer')

        # Get most common time of day of event occurence
        col = 'event_time_of_day_mode'
        df2 = horizons_df.groupby(groupby_cols)['time_of_day'].agg({
            col: pd.Series.mode
        }).reset_index().drop(columns=['level_2'])
        df = df.merge(df2, on=groupby_cols, how='outer')
        nominal_cols += [col]

    # Calculate adherence rate
    if 'day' in horizon:
        df2 = horizons_df.groupby(groupby_cols)['withinrange'].agg({
            'adherence_rate': 'max'
        }).reset_index()
    else:
        df2 = horizons_df.groupby(groupby_cols + ['study_day'])['withinrange'].max().reset_index() # Max will be 1 or 0
        df2 = df2.groupby(groupby_cols)['withinrange'].agg({
            'adherence_rate': lambda x: x.sum() / denom
        }).reset_index()
    
    df = df.merge(df2, on=groupby_cols, how='outer')
    
    # Help pandas since it doesn't process datetimes well and introduces duplicate entries on merges
    df = df.drop_duplicates(subset=groupby_cols)
    
    # Create a featureset from the resulting dataframe
    temporal_featuresets.append(features.Featureset(df=df,
                                                    name=horizon, #Intentional for now - using horizon as name
                                                    id_col=dataset.id_col,
                                                    horizon=horizon,
                                                    nominal_cols = nominal_cols))
    temporal_featuresets


# In[ ]:


# Sanity check
temporal_featuresets[0].df


# # Prediction

# In[ ]:


# Sanity check - this column should NOT be in the final set
# static_featuresets[0].df['total_days_8']


# In[ ]:


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


# ## Study 1: Predict Adherence from MEMS Data Only

# ### Tune number of lags

# In[ ]:


''' Test the model performance for a range of lags (number of previous inputs)
      and range of max_depths (since training with RF by default)
    max_depth exploration will help ensure we aren't overfitting.
'''
for t_feats in temporal_featuresets:
    models.tune_lags(t_feats)


# In[ ]:


results = pd.read_csv('results/tuned_lags/pred.csv')
results


# In[ ]:


for col in ['n_lags', 'accuracy', 'max_depth']:
    results[col] = pd.to_numeric(results[col])
results


# In[ ]:


results['specificity_loss'] = 1-results['specificity']
results


# In[ ]:


'''
Takeaways:
- Simpler models better for more granular time scales
- 2 lags with shallow tree (max_depth 1) best for study_day - overfitting worsens after that
- 1 lag with shallow tree (max_depth 1) best for study week
-- lets go with 2 to make it easy, for experiment - match with study day
- 4 lags with deeper tree (max depth 5) best for study_month
'''

for max_depth in range(1, 6):
    df = results[results['max_depth'] == max_depth]
    
    g = sns.lineplot(x='n_lags',y='specificity_loss', hue='featureset', style='type', data=df)
    g.set( ylim=(0, 0.5), title='Max Depth: ' + str(max_depth), ylabel='Specificity Loss')
    plt.show()


# In[ ]:


for t_feats in temporal_featuresets: 
    print(t_feats)


# ### Do prediction task

# In[ ]:


# # ----- Now predict using optimal number of lags for each horizon--- 
# n_lags = 2
# for t_feats in temporal_featuresets:    
#     # TODO: Set RF max-depth here after tuning lags
#     max_depth = -1
#     models = {
#         'LogisticR': LogisticRegression(solver='liblinear', random_state=random_state),
#         'RF': RandomForestClassifier(max_depth=max_depth, random_state=random_state),
#         'SVM': SVC(probability=True, random_state=random_state)
#     }
    
#     models.predict_from_mems(t_feats, n_lags, models)     


# ## Study 2: Predict Adherence from Demographic and Med Record Data

# In[ ]:


# TODO - build LSTM, following Esteban et al's approach

