from pathlib import Path

DATA_PATH = Path('data/')

OUTPUT_PATH_PRIMARY = Path('results/')
OUTPUT_PATH_LAGS = Path.joinpath(OUTPUT_PATH_PRIMARY, 'tuned_lags/')
OUTPUT_PATH_PRED = Path.joinpath(OUTPUT_PATH_PRIMARY, 'prediction_task/')
OUTPUT_PATH_LMM = Path.joinpath(OUTPUT_PATH_PRIMARY, 'lmm_task/')

# Standard time-based constants
SECONDS_IN_HOUR = 3600.0
DAYS_IN_WEEK = 7.0
DAYS_IN_MONTH = 30.0 # This is, of course, an avg estimate and is not precise

# Times of day of MEMS use
TIME_OF_DAY_PROPS ={
    'bins': [-1, 6, 12, 18, 24],
    'labels': ['early_morning', 'morning', 'afternoon', 'evening']
}

# Standard time horizons over which we want to predict
# TARGET_HORIZONS = ['study_day', 'study_week', 'study_month']
TARGET_HORIZONS = ['study_day', 'study_week']

# Threshold under which participants are considered to be nonadherent
ADHERENCE_THRESHOLD = 0.8

# Specify columns to rename - want this project to be human-readable!
RENAMINGS = {
    'demographics': {
        'EDU_RECODE':'education', 
        'Country': 'birth_country',
        'A_DEMO1': 'age',
        'A_DEMO2': 'hispanic_latina',
        'A_DEMO5': 'marital_status',
        'A_DEMO6': 'employment',
        'A_DEMO8': 'income',
        'A_DEMO10': 'years_in_usa',
        'A_DEMO11': 'primary_language',
        'A_DEMO_12': 'num_people_in_household',
        'A_DEMO13YN': 'take_meds_regularly',
        'A_DEMO31': 'race_white',
        'A_DEMO32': 'race_black',
        'A_DEMO33': 'race_asian',
        'A_DEMO34': 'race_native_hawaiian_pacific_islander',
        'A_DEMO35': 'race_native_american',
        'A_DEMO36': 'race_other', # freetext
        'A_DEMO37': 'race_prefer_not_to_answer',
        'A_DEMO38': 'race_unsure',
    },
    'medical': { # From medical record abstraction forms (pre/post)
        'stage_recoded': 'stage',
        'A_MR1': 'pre_insurance_status',
        'A_MR3': 'pre_dx_date',
        
        # --- Treatments received prior to enrollment --- 
        'A_MR4_YN': 'pre_radiation',
        'A_MR5_YN': 'pre_chemo',
        'A_MR6_YN': 'pre_surgery',
        'A_MR7_YN': 'pre_reconstructive_surgery',
         #  --- Drugs prescribed prior to enrollment --- 
        'A_MR9_tamox_YN': 'pre_tamoxifen',
        'A_MR9_ralox_YN': 'pre_raloxifene',
        'A_MR9_torem_YN': 'pre_toremifene',
        'A_MR9_anas_YN': 'pre_anastrazole',
        'A_MR9_exem_YN': 'pre_exemestane',
        'A_MR9_let_YN': 'pre_letrozole',
        'A_MR9_gose_YN': 'pre_goserelin',
        #  --- Comorbid conditions prior to enrollment --- 
        'A_MR10_a': 'pre_myocardial_infarction', 
        'A_MR10_b': 'pre_cong_heart_failure',
        'A_MR10_c': 'pre_peripheral_vascular_disease',
        'A_MR10_d': 'pre_cerebrovascular_disease',
        'A_MR10_e': 'pre_dementia',
        'A_MR10_f': 'pre_chromic_pulm_disease',
        'A_MR10_g': 'pre_conn_tissue_disease',
        'A_MR10_h': 'pre_peptic_ulcer_disease',
        'A_MR10_i': 'pre_liver_disease_mild',
        'A_MR10_j': 'pre_diabetes',
        'A_MR10_k': 'pre_hemiplegia',
        'A_MR10_l': 'pre_renal_disease',
        'A_MR10_m': 'pre_diabetes_and_organdamage',
        'A_MR10_n': 'pre_leukemia',
        'A_MR10_o': 'pre_lymphoma',
        'A_MR10_p': 'pre_liver_disease_mod_severe',
        'A_MR10_q': 'pre_met_tumor',
        'A_MR10_r': 'pre_aids',
        'A_MR10_s': 'pre_cancer_other',
        'A_MR10_t': 'pre_arthritis',
        'A_MR10_u': 'pre_mental_illness',
    }
}

# Codebook for categoricals
CODEBOOK = {
    'education': {
        1: 'no_education',
        2: 'grade_school',
        3: 'some_high_school',
        4: 'high_school_grad',
        5: 'some_college',
        6: 'college_grad',
        7: 'grad_school'
    },
    'marital_status': {
        1: 'single',
        2: 'married',
        3: 'divorced',
        4: 'widowed'
    },
    'employment': {
        0: 'not_employed',
        1: 'part_time',
        2: 'full_time'
    },
    'income': {
        0: 'no_income',
        1: '1_to_9999',
        2: '10000_to_19999',
        3: '20000_to_29999',
        4: '30000_to_39999',
        5: '40000_to_49999',
        6: '50000_to_74999',
        7: '75000_to_99999',
        8: '100000_or_more'
    },
    'birth_country': {
        1: 'USA',
        2: 'Mexico', 
        3: 'Other'
    }, 
    'primary_language': {
        1: 'English',
        2: 'Spanish',
        3: 'Portuguese', 
        4: 'Other'
    }
}

SCORE_PREFIXES = ['A_', 'B_', 'C_']

''' Many measures were administered at different time points (e.g., pre, mid-study, and post),
  so we'll need to use a regex to find all columns for each score category

We'll also want to store things like the max score for each measure, which will help us
  normalize the vals later'''
SCORES = {
    'BACS': {
        'semantic_label': 'Barriers to Care Scale',
        'suffix': 'BACS',
        'max_val': 140, 
        'precalculated': False,
        'subscale_include': False
    },
    'BCPT': {
        'semantic_label': 'Breast Cancer Prevention Trial Symptom Checklist',
        'suffix': 'BCPT',
        'max_val': 168, 
        'precalculated': False,
        'subscale_include': False
    },
    'BCSK': {
        'semantic_label': 'Breast Cancer Survivorship Knowledge',
        'suffix': 'BCSK',
        'max_val': 32, 
        'precalculated': False,
        'subscale_include': False
    },
    'BMQ': {
        'semantic_label': 'Beliefs About Medicines Questionnaire',
        'suffix': 'BMQ',
        'max_val': 50, 
        'precalculated': False,
        'subscale_include': False
    },
    'CASE': {
        'semantic_label': 'CASE',
        'suffix': 'CASE',
        'max_val': 48, 
        'precalculated': False,
        'subscale_include': False
    },
    'DECREG': {
        'semantic_label': 'Decision Regret Scale',
        'suffix': 'DECREG',
        'max_val': 25, 
        'precalculated': False,
        'subscale_include': False
    },
    'FACIT_SP': {
        'semantic_label': 'FACIT-SP',
        'suffix': 'FACITSP',
        'max_val': 48, 
        'precalculated': False,
        'subscale_include': False
    },
    'FACT_B': {
        'semantic_label': 'FACT-B',
        'suffix': 'FACTB',
        'max_val': 148, 
        'precalculated': True,
        'subscale_include': False
    },
    'FACT_BC': {
        'semantic_label': 'FACT Breast Cancer Subscale',
        'suffix': 'BC',
        'max_val': 40, 
        'precalculated': True,
        'subscale_include': False # This is a FACT-G subscale, but we're more interested in FACT-B vs FACT-G
    },
    'FACT_G_PWB': {
        'semantic_label': 'FACT-G Physical Well-Being Subscale',
        'suffix': 'PWB',
        'max_val': 28, 
        'precalculated': True,
        'subscale_include': True
    },
    'FACT_G_SWB': {
        'semantic_label': 'FACT-G Social Physical Well-Being Subscale',
        'suffix': 'SWB',
        'max_val': 28,
        'precalculated': True,
        'subscale_include': True
    },
    'FACT_G_EWB': {
        'semantic_label': 'FACT-G Emotional Physical Well-Being Subscale',
        'suffix': 'EWB',
        'max_val': 28, 
        'precalculated': True,
        'subscale_include': True
    },
    'FACT_G_FWB': {
        'semantic_label': 'FACT-G Functional Physical Well-Being Subscale',
        'suffix': 'FWB',
        'max_val': 28, 
        'precalculated': True,
        'subscale_include': True
    },
    'FACT_G': {
        'semantic_label': 'FACT-G',
        'suffix': 'FACTG',
        'max_val': 108, 
        'precalculated': True,
        'subscale_include': False
    },
    'MDASI': {
        'semantic_label': 'MD Anderson Symptom Inventory',
        'suffix': 'MDASI',
        'max_val': 190, 
        'precalculated': False,
        'subscale_include': False
    },
    'MASES': {
        'semantic_label': 'Medication Adherence Self-Efficacy Scale',
        'suffix': 'MASES',
        'max_val': 210, 
        'precalculated': False,
        'subscale_include': False
    },
#     'MEDAD': {
#         'semantic_label': 'Medication Adherence Scale',
#         'suffix': 'MEDAD',
#         'max_val': -1, 
#         'precalculated': False
#     },
    'PEARL': {  # I think Kristi said don't use this
        'semantic_label': 'Pearlin Mastery Scale',
        'suffix': 'PEARL',
        'max_val': 35, 
        'precalculated': False,
        'subscale_include': False
    },
    'PTGI': {
        'semantic_label': 'Posttraumatic Growth Inventory',
        'suffix': 'PTGI',
        'max_val': 50, 
        'precalculated': False,
        'subscale_include': False
    },
    'PSAT': {
        'semantic_label': 'Patient Satisfaction',
        'suffix': 'PSAT',
        'max_val': 85, 
        'precalculated': False,
        'subscale_include': False
    },
    'PSS': {
        'semantic_label': 'Perceived Stress Scale',
        'suffix': 'PSS',
        'max_val': 70, 
        'precalculated': False,
        'subscale_include': False
    },
    'PSUSP': {
        'semantic_label': 'Perceived Susceptibility Scale',
        'suffix': 'PSUSP',
        'max_val': 15, 
        'precalculated': False,
        'subscale_include': False
    },
    'SS': {  
        'semantic_label': 'Krause and Borwaski-Clark Social Support Scale',
        'suffix': 'SS',
        'max_val': 42, 
        'precalculated': False,
        'subscale_include': False
    },

}
