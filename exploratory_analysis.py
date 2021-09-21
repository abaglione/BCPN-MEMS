#!/usr/bin/env python
# coding: utf-8

# # Loading

# In[23]:


# IO
from pathlib import Path
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

# Utility Libraries
import math
from datetime import datetime
import re
import csv
import itertools

# Data Processing
import pandas as pd
import numpy as np

# Predictive Analytics
import statsmodels.stats.api as sms
from sklearn.feature_selection import VarianceThreshold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from bcpn_pipeline import data, features, models, consts
import shap

# Viz
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
from matplotlib.dates import DateFormatter
from matplotlib.cbook import boxplot_stats
import matplotlib.dates as mdates
import matplotlib.transforms as mtrans
import seaborn as sns
sns.set_style("whitegrid")

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
# plt.rcParams.update({'figure.facecolor': [1.0, 1.0, 1.0, 1.0]})

# configure autoreloading of modules
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[24]:


# Load the data
# datafile = Path("data/final_merged_set_v6.csv")
datafile = Path("/mnt/c/Users/ab5bt/Box Sync/Research/UVA/Medication Adherance/MEMS dataset/final_merged_set_v6.csv")
df = pd.read_csv(datafile, parse_dates=False)
df.head()


# # Data Cleaning & Feature Engineering
# Thank you to Jason Brownlee
# https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/

# In[25]:


# Instantiate a Dataset class
dataset = data.Dataset(df, id_col = 'PtID')
dataset


# In[26]:


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

# In[27]:


''' 
 Organize the candidate features into useful categories for later reference
 A bit tedious, but helpful 
'''

# Set our excluded features, before anything else
excluded = ['percentMEMS8', 'total_days_8']  # Related to definition of adherence
dataset.update_feature_categories({
    'demographics': [v for v in consts.RENAMINGS['demographics'].values()
                     if v in dataset.df.columns] + ['race'], #add the new, single race col
    'study_behavior': [col for col in ['DateEnroll', 'Group', 'complete_4', 
                                       'complete_8', 'memsuse', 'deceased',
                                       'day_miss_fromB', 'day_miss_from7'] 
                       if col in dataset.df.columns],
    'medical': [v for v in consts.RENAMINGS['medical'].values() if v in dataset.df.columns] + \
               [col for col in ['early_late', 'diagtoenroll'] 
                if col in dataset.df.columns]
})


# In[28]:


''' This dataset has several repeated measures for validated instruments, 
such as the FACTB

Columns for repeated measures for the same instrument share a suffix (e.g., '_FACTB')
Use regex to populate the `scores` category subdictionary quickly, using these suffixes
''' 

# TODO: Fix scores so that we only have one column per score
# incorporate the shift halfway through the study (i.e. midpoint assessments)

for k,v in consts.SCORES.items():
    ''' Handle special case of BCPT before doing anything else '''
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
    
    '''Some measures weren't precalculated. Let's fix this 
    We'll only focus on time point A, since it doesn't make sense to make predictions using
    future scores!
    ''' 
    if v['precalculated'] == False:
        
        ''' For the baseline time point, get the aggregate score and add it to the dataset
        as a new column'''
        prefix = 'A'
        score_cols = list(
            dataset.df.filter(regex='^' + prefix + v['suffix'] + '\d*').columns
        )
        
        dataset.df[prefix + v['suffix']] = dataset.df[score_cols].sum(axis=1)
    '''We'll include this new column as a feature shortly'''
   
    dataset.update_feature_categories({
        'scores': [prefix + v['suffix']]
    })


# In[29]:


''' Create a catch-all category of remaining features, to ensure we got everything '''
dataset.update_feature_categories({
    'other': [col for col in dataset.df.columns 
              if col not in list(itertools.chain(*dataset.feature_categories.values())) # exclude anything already in the list
              and not any(prefix in col for prefix in ['A_', 'B_', 'C_']) # exclude individual score cols
              and 'date' not in col 
              and col not in dataset.id_col
              and col not in excluded
             ]
})


# In[30]:


dataset.feature_categories


# ### Generate new features

# In[31]:


''' Create new columns for several demographic and medical variables
Be sure we update the feature categories dictionary '''

demog_drug_cols = [col for col in dataset.df.columns if 'A_DEMO13DRUG' in col]
newcol = 'DEMOG_numdrugs'
dataset.df[newcol] = dataset.df[demog_drug_cols].count(axis=1)
dataset.update_feature_categories({'demographics': [newcol]})

post_exam_cols = [col for col in dataset.df.columns if 'C_MR5_date' in col]
dataset.df[post_exam_cols] = dataset.df[post_exam_cols].apply(
    lambda x: pd.to_datetime(x, errors='coerce')
)
newcol = 'C_numexams'
dataset.df[newcol] = dataset.df[post_exam_cols].count(axis=1)
dataset.update_feature_categories({'medical': [newcol]})

''' Thank you @benvc!
https://stackoverflow.com/questions/54367491/calculate-average-of-days-between-a-list-of-dates
'''

# TODO: Dates aren't necessarily in order. Ask Kristi if this is a data entry issue or 
# An ordering issue?
newcol = 'mean_days_betw_exams'
dataset.df[newcol] = dataset.df[post_exam_cols].apply(
    lambda x: features.mean_days_between_dates(x),
    axis=1
)
dataset.update_feature_categories({'medical': [newcol]})

# Ensure everything looks good
print(dataset.df['DEMOG_numdrugs'].head())
print(dataset.df['C_numexams'].head())
print(dataset.df['mean_days_betw_exams'].head())


# ### Create featuresets

# In[32]:


static_featuresets = list()
categories = list(dataset.feature_categories.keys())
fact_subscales = ['A' + v['suffix'] for k,v in consts.SCORES.items() if v['subscale_include'] == True]
drop_pairs = [
    ('FACTB Subset', ['A_FACTG'] + fact_subscales), # FACT-B only
    ('FACTB Subscales Subset', ['A_FACTB', 'A_FACTG']), # FACT-B subscales only
    ('FACTG Subset', ['A_FACTB'] + fact_subscales) # FACT-G only
]

print(fact_subscales)

df = dataset.build_df_from_feature_categories(categories)
static_featuresets.append(features.Featureset(df=df, name=name + ' - ' + 'all_scores', id_col = dataset.id_col))

# Create three further subsets related to FACT scores
for (subset_name, drop_cols) in drop_pairs:
    df2 = df.drop(columns=drop_cols) # Returns a copy
    static_featuresets.append(features.Featureset(df=df2, name=name + ' - ' + subset_name, 
                                                  id_col = dataset.id_col))
        
static_featuresets


# ## Dynamic (Temporal) Features

# Extract temporal features by converting main dataset's df from wide-form to long-form.

# In[33]:


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

df = pd.concat(rows, axis=0)

# Create combined datetime column
df['datetime'] = df.apply(
    lambda x: features.get_datetime_col(x), axis=1
)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# Fix dtypes
df[['withinrange', 'num_times_used_today']] = df[['withinrange', 'num_times_used_today']].fillna(0).astype(int)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['interval'] = pd.to_timedelta(df['interval']) # Handle NaT intervals for first day?

'''Drop rows with an empty date column.
  Do NOT drop empty time columns - may have dates where it is recorded that the patient
  did not use the cap. So, would have a date but no time. Need this info to calculate
  additional stats later
''' 
df.dropna(subset=['date'], inplace=True)

# Drop duplicates - these must have been introduced with the melt and with how Kristi's original data was structured
df.drop_duplicates(inplace=True)

# Remove observations that occurred before a subject's enrollment date
df = df.loc[df['DateEnroll'] < df['date']]

# Restrict to 210 MEMS days (not necessarily study days), per Kristi's documentation
df = df[df['MEMS_day'] <= 210] 

df


# In[34]:


# Sanity check - Validate that we calculated days of adherence correctly
df2 = df.groupby([dataset.id_col, 'MEMS_day'])['withinrange'].max().reset_index()
df2 = df2.groupby(dataset.id_col)['withinrange'].sum().reset_index()
df2 = df2.merge(dataset.df[[dataset.id_col, 'total_days_8']])
df2.head(10)
# Check!


# ### Generate new features

# In[35]:


# Add binary indicator of any usage (not just number of times used) on a given day
df['used_today'] = df['num_times_used_today'].apply(
    lambda x: 1 if x > 0 else 0
)

'''Generate epochs of interest (time of day, weekday, day/month of study, etc)
   'time_of_day' category gets automatically encoded as a Categorical
''' 
time_of_day_props = {
    'bins': [-1, 6, 12, 18, 24],
    'labels': ['early_morning', 'morning', 'afternoon', 'evening']
}
df = features.get_epochs(df, 'DateEnroll', 'PtID',
                         time_of_day_props['bins'], 
                         time_of_day_props['labels'])
df.head()

# TODO - figure out a way to use the times of day? Not currently being used


# In[36]:


# Exclude first month (ramp-up period during which time users were getting used to the MEMS caps)
df = df[df['study_month'] > 0]


# In[37]:


temporal_featuresets = list()
'''Group by our desired epoch and add standard metrics such as mean, std
'''

# TODO - add ability to calculate adherence metrics by month, extend epochs to include months
# Will need a way to find days in a given month...

for epoch in ['study_day', 'study_week']:
    groupby_cols = [dataset.id_col, epoch]
    
    temporal_feats = df.groupby(groupby_cols)['datetime'].agg({
        'n_events': 'count'
    }).reset_index()
    
    if 'day' in epoch:
        df2 = df[groupby_cols + ['is_weekday']]
        temporal_feats = temporal_feats.merge(df2, on=groupby_cols)
    else:
        df2 = features.calc_standard_temporal_metrics(df, groupby_cols, 'datetime')
        temporal_feats = temporal_feats.merge(df2, on=groupby_cols)

        # Calculate avg and standard deviation of number of times used
        df2 = df.groupby(groupby_cols + ['study_day'])['num_times_used_today'].max().reset_index()
        df2 = df.groupby(groupby_cols)['num_times_used_today'].agg({
            'num_daily_events_mean': lambda x: x.sum() / consts.DAYS_IN_WEEK
        }).reset_index()
        temporal_feats = temporal_feats.merge(df2, on=groupby_cols)

        # Get most common time of day of event occurence
        df2 = df.groupby(groupby_cols)['time_of_day'].agg({
            'event_time_of_day_mode': pd.Series.mode
        }).reset_index().drop(columns=['level_2'])
        temporal_feats = temporal_feats.merge(df2, on=groupby_cols)

    # Calculate adherence rate
    if 'day' in epoch:
        df2 = df.groupby(groupby_cols)['withinrange'].agg({
            'adherence_rate': 'max'
        }).reset_index()
    else:
        df2 = df.groupby(groupby_cols + ['study_day'])['withinrange'].max().reset_index() # Max will be 1 or 0
        df2 = df2.groupby(groupby_cols)['withinrange'].agg({
            'adherence_rate': lambda x: x.sum() / consts.DAYS_IN_WEEK
        }).reset_index()
    
    temporal_feats = temporal_feats.merge(df2, on=groupby_cols)

    temporal_featuresets.append(features.Featureset(df=temporal_feats,
                                                    name=epoch, #Intentional for now - using epoch as name
                                                    id_col=dataset.id_col,
                                                    epoch=epoch
                                                   )
                               )
temporal_featuresets


# # Prediction

# In[38]:


# Sanity check - this column should NOT be in the final set
# static_featuresets[3].df['total_days_8']


# ## Run prediction tasks

# ### Tune number of lags

# In[39]:


target_col = 'adherent'

# Test the performance for a range of lags (number of previous inputs)
for temporal_feats in temporal_featuresets:
    
    #Convert adherence_rate into a binary indicator of adherence
    temporal_feats.df[target_col] = temporal_feats.df['adherence_rate'].apply(
        lambda x: 1 if x > consts.ADHERENCE_THRESHOLD else 0
    )
    # Drop the original column
    temporal_feats.df.drop(columns=['adherence_rate'], inplace=True)

    # Set the target col
    temporal_feats.target_col = target_col
    
#     for n_lags in range(2, 16):
#         print('For ' + str(n_lags) + ' lags...')
#         results = []

#         #Perform final encoding, scaling, etc
#         all_feats = temporal_feats.prep_for_modeling(n_lags)
        
#         # Ensure we got a lagged series as expected
#         print(all_feats.df)
        
#         # Do 10 runs per lag
#         print('Running prediction 10 times...')
#         for j in range(0,10):

#             # Do our actual predictions
#             # Use defaults in this go-round, rather than gridsearch
#             res = models.predict(all_feats, n_lags, classifiers=['RF'], optimize=False, importance=False)
#             results.append(res)

#         lag_results = pd.concat(results, axis=0)
#         lag_results.to_csv('results/prelim_pred_results_' + all_feats.name + '_' + str(n_lags) + '_lags.csv',index=False)


# In[ ]:


# import glob
# all_files = glob.glob("results/*.csv")
# all_files

# li = []

# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     li.append(df)

# results = pd.concat(li, axis=0, ignore_index=True)
# results.head()


# In[ ]:


# sns.lineplot(x='n_lags',y='test_accuracy', hue='featureset', data=results)


# In[42]:


target_col = 'adherent'
all_results = []         
n_lags = 6

# TODO: Change so that we feed only the X most predictive features (according to SHAP scores)
# from the random forest into the SVM
# Saw this approach in a diff paper...just need to remember where
# Will likely help performance

# ----- Now predict using optimal number of lags --- 
    
for temporal_feats in temporal_featuresets:

    # Get a lagged featureset
    temporal_feats_lagged = temporal_feats.prep_for_modeling(n_lags)

    # Predict from only the temporal features first
    res = models.predict(temporal_feats_lagged, n_lags, classifiers=['SVM'], optimize=True)
    print(res)
    all_results.append(res)       

    i = 0
    for static_feats in static_featuresets:

        # Now get results for both static and dynamic features
        static_feats.prep_for_modeling()   
        all_feats = temporal_feats_lagged.create_combined_featureset(fs=static_feats)

        ''' Don't forget to check for multicollinearity in the newly-combined set!
        This step is automatically-handled during prep_for_modeling
        However, we DON'T want to run prep_for_modeling again since it would create more lags.
        Yes this is clunky. Yes I'm tired...
        '''
        all_feats.handle_multicollinearity()
        res = models.predict(all_feats, n_lags, optimize=True)
        all_results.append(res)                                 


# In[ ]:


shap_scores = pickle.load(open('feature_importance/shap_RF_' + str(n_lags) + '_lags_optimized.ob', 'rb'))
X_test = pd.read_pickle('feature_importance/X_test_RF_' + str(n_lags) + '_lags.ob')
X_test.columns = [x.replace('_', ' ').capitalize() if 'A_' not in x else x.replace('A_', '') for x in X_test.columns]
shap.summary_plot(shap_scores[1], X_test, show=False)
fig = plt.gcf()
fig.set_size_inches(12.5, 8.5)
plt.savefig('feature_importance/RF.png')
plt.show()

