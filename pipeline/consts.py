'''
Specify columns to rename - want this project to be human-readable!

Note - I manually created any needed dummy vars in SPSS and named them what I wanted

Any dummy vars included in this renaming dictionary are vars leftover from Kristi's analysis that
weren't renamed in place 
''' 
RENAMINGS = {
    'demographics': {
        'EDU_RECODE':'education', 
        'Country': 'country',
        'A_DEMO1': 'age',
        'A_DEMO31': 'race_white',
        'A_DEMO32': 'race_black',
        'A_DEMO33': 'race_asian',
        'A_DEMO34': 'race_native_hawaiian_pacific_islander',
        'A_DEMO35': 'race_native_american',
        'A_DEMO36': 'race_other',
        'A_DEMO37': 'race_prefer_not_to_answer',
        'A_DEMO38': 'race_unsure',
        'A_DEMO2': 'hispanic_latina',
        'A_DEMO10': 'years_in_usa',
        'A_DEMO11': 'primary_lang',
        'A_DEMO13YN': 'take_meds_regularly',
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
        # ---- Procedures recommended / received since enrollment --- 
        'C_MR3': 'post_num_exams_recommended',
        'C_MR4': 'at_least_one_exam',
        'C_MR5': 'post_all_exams_obtained',
        'C_MR6': 'post_num_mammograms_recommended',
        'C_MR7_RECODED': 'post_any_mammograms_received',
        'C_MR7_YN_RECODED': 'post_all_mammograms_received',
        # ---- Drugs prescribed since enrollment ---
        'C_MR9_tamox_YN': 'post_tamoxifen',
        # Note that `evist` and `ralox` prefixes both refer to same med
        'C_MR9_evist_YN': 'post_raloxifene', 
        # Same note as above; different prefixes used here, but refer to same respective meds
        'C_MR9_fares_YN': 'post_toremifene',
        'C_MR9_arim_YN': 'post_anastrazole',
        'C_MR9_aroma_YN': 'post_exemestane',
        'C_MR9_femara_YN': 'post_letrozole',
        'C_MR9_zola_YN': 'post_goserelin', 
        'C_MR10': 'post_changed_meds',
        'C_MR12': 'post_side_effects',
        'C_MR18': 'post_num_appts_scheduled',
        'C_MR19': 'post_num_appts_canceled_by_patient',
        'C_MR20': 'post_num_appts_missed',
        # C_MR21 through C_MR23 definitely have low variance - not going to bother to rename / include
        'C_MR24_MRI': 'post_received_mri',
        'C_MR24_US': 'post_received_ultrasound',
        'C_MR24_BI': 'post_received_biopsy',
        'C_MR24_CT': 'post_received_ct_scan',
        'C_MR24_BS': 'post_received_bone_scan',
        'C_MR24_AS': 'post_received_addnl_surgery',
        'C_MR24_AR': 'post_received_addnl_radiation',
        'C_MR24_GT': 'post_received_genetic_couns_test',
        'C_MR24_GE': 'post_received_gyn_exam',
        'C_MR24_Otherbreastsurgery': 'post_received_other_breast_surgery',
        # C_MR25 through C_MR27 definitely have low variance - not going to bother to rename / include
    }
}

''' 
Many measures were administered at different time points (e.g., pre, mid-study, and post),
  so we'll need to use a regex to find all columns for each score category

We'll also want to store things like the max score for each measure, which will help us
  normalize the vals later
  
'''
SCORES = {
    'BACS': {
        'semantic_label': 'Barriers to Care Scale',
        'suffix': '_BACS',
        'max_val': 140, 
        'precalculated': False
    },
    'BCPT': {
        'semantic_label': 'Breast Cancer Prevention Trial Symptom Checklist',
        'suffix': '_BCPT',
        'max_val': 168, 
        'precalculated': False
    },
    'BCSK': {
        'semantic_label': 'Breast Cancer Survivorship Knowledge',
        'suffix': '_BCSK',
        'max_val': 32, 
        'precalculated': False
    },
    'BMQ': {
        'semantic_label': 'Beliefs About Medicines Questionnaire',
        'suffix': '_BMQ',
        'max_val': 50, 
        'precalculated': False
    },
    'CASE': {
        'semantic_label': 'CASE',
        'suffix': '_CASE',
        'max_val': 48, 
        'precalculated': False
    },
    'DECREG': {
        'semantic_label': 'Decision Regret Scale',
        'suffix': '_DECREG',
        'max_val': 25, 
        'precalculated': False
    },
    'FACIT_SP': {
        'semantic_label': 'FACIT-SP',
        'suffix': '_FACITSP',
        'max_val': 48, 
        'precalculated': False
    },
    'FACT_B': {
        'semantic_label': 'FACT-B',
        'suffix': '_FACTB',
        'max_val': 148, 
        'precalculated': True
    },
    'FACT_BC': {
        'semantic_label': 'FACT Breast Cancer Subscale',
        'suffix': '_BC',
        'max_val': 40, 
        'precalculated': True
    },
    'FACT_G_PWB': {
        'semantic_label': 'FACT-G Physical Well-Being Subscale',
        'suffix': '_PWB',
        'max_val': 28, 
        'precalculated': True
    },
    'FACT_G_SWB': {
        'semantic_label': 'FACT-G Social Physical Well-Being Subscale',
        'suffix': '_SWB',
        'max_val': 28,
        'precalculated': True
    },
    'FACT_G_EWB': {
        'semantic_label': 'FACT-G Emotional Physical Well-Being Subscale',
        'suffix': '_EWB',
        'max_val': 28, 
        'precalculated': True
    },
    'FACT_G_FWB': {
        'semantic_label': 'FACT-G Functional Physical Well-Being Subscale',
        'suffix': '_FWB',
        'max_val': 28, 
        'precalculated': True
    },
    'FACT_G': {
        'semantic_label': 'FACT-G',
        'suffix': '_FACTG',
        'max_val': 108, 
        'precalculated': True
    },
    'MDASI': {
        'semantic_label': 'MD Anderson Symptom Inventory',
        'suffix': '_MDASI',
        'max_val': 190, 
        'precalculated': False
    },
    'MASES': {
        'semantic_label': 'Medication Adherence Self-Efficacy Scale',
        'suffix': '_MASES',
        'max_val': 210, 
        'precalculated': False
    },
#     'MEDAD': {
#         'semantic_label': 'Medication Adherence Scale',
#         'suffix': '_MEDAD',
#         'max_val': -1, 
#         'precalculated': False
#     },
    'PEARL': {  # I think Kristi said don't use this
        'semantic_label': 'Pearlin Mastery Scale',
        'suffix': '_PEARL',
        'max_val': 35, 
        'precalculated': False
    },
    'PTGI': {
        'semantic_label': 'Posttraumatic Growth Inventory',
        'suffix': '_PTGI',
        'max_val': 50, 
        'precalculated': False
    },
    'PSAT': {
        'semantic_label': 'Patient Satisfaction',
        'suffix': '_PSAT',
        'max_val': 85, 
        'precalculated': False
    },
    'PSS': {
        'semantic_label': 'Perceived Stress Scale',
        'suffix': '_PSS',
        'max_val': 70, 
        'precalculated': False
    },
    'PSUSP': {
        'semantic_label': 'Perceived Susceptibility Scale',
        'suffix': '_PSUSP',
        'max_val': 15, 
        'precalculated': False
    },
    'SS': {  
        'semantic_label': 'Krause and Borwaski-Clark Social Support Scale',
        'suffix': '_SS',
        'max_val': 42, 
        'precalculated': False
    },

}
