#%% import packages
# Import Python libraries
import sys
import csv
import itertools
import time
from datetime import datetime
import os
import csv
import numpy as np
import pandas as pd
import random
import ast
import logging
sys.path.append('../../scripts/')
import logging
import json
import itertools
from pathlib import Path
from Setup_SCWSM_Option_Analysis_CST import simSetup
print('import packages')

#%% Define climate and other scenarios to run
def setup_random_sims(num_sims, random_seed):

    # set up climate scenarios
    random.seed(random_seed)
    real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]  # [1270] # 1270, 1956, 1987, 2770,
    dT_All = np.arange(0, 6, 1)
    dP_All = np.arange(60, 121, 10)
    dCV_All = ['1.0', '1.1', '1.2']
    demand_All = ['Baseline', 'Low', 'High']

    # get random combinations of inputs
    combinations = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))
    random_combinations = random.sample(combinations, num_sims)

    return random_combinations

def setup_climate_sims(real_All, dT_All, dP_All, dCV_All, demand_All):

    # get combinations of inputs
    combinations = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))

    print('all scenario combinations: {}'.format(combinations))

    return combinations

# Purpose: post-process the results dataframe from Pywr
# Inputs:
#   df - dataframe from Pywr model instance
# Outputs:
#   df_processed - processed and cleaned up df
def post_process_results(df):
    # Take out the first empty row of the dataframe and rename the index
    df.columns = df.columns.get_level_values(0).tolist()
    df.index.name = 'Date'
    df.index = df.index.to_timestamp()
    # Sort the column by alphabetic order, and save the dataframe into a.csv file
    df_processed = df.reindex(sorted(df.columns), axis=1)
    return df_processed

#%% Define function for simulation model
# main difference from function for optimization is saving more simulation outputs
# inputs:
#   a list of decision variables- rof threshold, inf order
#   filepath for saving results
# outputs/saved files:
#   df_results: dataframe of daily result from pywr
#   df_cashflow: table of monthly cashflow model results
#   df_timetracker: table of time tracker results
#   arr_hh_data: array of household-level data

# run simulation function- v2 where we input climate characteristics
def sim_model_run_v2(decision_vars, filepath, real_All, dT_All, dP_All, dCV_All, demand_All, filepath_SA, name_add=''):

    # setup parameters for climate combinations
    #num_sims = 1
    #random_seed = 4
    # get climate combinations
    scenario = setup_climate_sims(real_All, dT_All, dP_All, dCV_All, demand_All)
    print(scenario)
    num_sims = len(scenario)

    # save climate scenario combinations
    headers = ['real', 'dT', 'dP', 'dCV', 'demand']
    # Save the list to a CSV file
    # with open(filepath + 'test_random_combinations_03Sept_NA.csv', 'w', newline='') as file:
    with open(filepath + 'climate_scenarios_21Nov2025.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(scenario)

    # set up parameters for versions
    version_parent = 'SCWSM-Option_Analysis'
    version = 'SCWSM-SimOpt_Test'
    options = [None]

    # run one simulation at a time (I think I want to parallelize this)
    start_time = time.time()
    print('num sims: {}'.format(num_sims))
    for i in range(num_sims):
        print('simulation: {}'.format(str(i+1)))
        print(scenario[i])
        # create an instance of the model
        modelSetup = simSetup(scenario[i][0], scenario[i][1], scenario[i][2], scenario[i][3], options, filepath_SA,
                              scenario[i][4], decision_vars)
        model = modelSetup.m  # convert from simSetup class to pywr model object

        # run the model
        model.run()

        # post-processing
        # results dataframe
        df_results = post_process_results(model.to_dataframe())
        # cashflow df
        df_cashflow = model.parameters['cashflow_model'].df_cashflow
        # add total demands by tier to df_cashflow
        num_tiers = model.parameters['cashflow_model'].num_tiers
        for j in range(num_tiers):
            print('tier: {}'.format(j))
            df_cashflow['demand_t{}'.format(j+1)] = model.parameters['previous_time_step_demand'].arr_demand_by_tier[:, j]

        print('after running tiers')
        # df time tracker
        df_time_tracker = model.parameters['cashflow_model'].df_time_tracker

        # all household data
        arr_hh = model.parameters['santa_cruz_demand_MGD'].arr_hh_data

        # save results
        # dataframes to save- df_cashflow, df_rates, df_results,
        dataframes = [df_results,  df_cashflow, df_time_tracker]
        filenames = ['df_results_{}P{}T{}_dCV{}_real{}_demand{}.csv'.format(name_add, scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4]),
                     'df_cashflow_{}P{}T{}_dCV{}_real{}_demand{}.csv'.format(name_add, scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4]),
                     'df_time_tracker_{}P{}T{}_dCV{}_real{}_demand{}.csv'.format(name_add, scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4])]
        print(filenames)
        # Iterate and save each DataFrame as a CSV
        for df, filename in zip(dataframes, filenames):
            df.to_csv(filepath + filename, index=True)

        filename = 'arr_hh_data_{}P{}T{}_dCV{}_real{}_demand{}.npy'.format(name_add, scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4])
        np.save(filepath+filename, arr_hh)

        print('saved data')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Model ran in {elapsed_time} seconds')
    return model


#%% Set up json files dynamically
# list of options to loop over
PriceElasticity = ['High', 'Low'] # -0.5, 0
MultiFamily = ['High', 'Low']
SF_Demands_Multiplier = ['High', 'Low'] # 1.15, 0.85
SF_Households = ['High', 'Low'] #23,507 and 19,233
SF_Income = ['High', 'Low'] # 36% HI, 28% LI

# initialize file name
household_data_file = 'resampled_income_data_30Nov2024.csv'

# output directory
outdir = Path("../model_assumptions_and_scenarios/sensitivity_analysis_DemandDemographics/") 
outdir.mkdir(exist_ok=True)

# loop through combinations to set up json files
# --- loop over all combinations ---
for pe, dem, house, income in itertools.product(PriceElasticity, SF_Demands_Multiplier, SF_Households, SF_Income):
    # build filename based on parameters
    filename = f"SA_PE={pe}_Dem={dem}_House={house}_Income={income}.json"
    filepath = outdir / filename

    # SF Demands Multiplier
    if dem == 'High':
        mult = 1.15
    elif dem == 'Low':
        mult = 0.85
    else:
        mult = 1.0

    # price elasticity
    if pe == 'High':
        dcc_coefs_file = 'DCCcoefs_elast=-0.5.csv'
    elif pe == 'Low':
        dcc_coefs_file = 'DCCcoefs_elast=0.csv'
    else:
        dcc_coefs_file = 'DCCcoefs.csv'

    # multi-family demands scenario
    # skip this for json file

    # single-family income distribution and # of households
    household_data_file = 'resampled_income_data_{}_hhs_{}.csv'.format(income, house)

    # build JSON dictionary
    data = {
        "factor_demand_multiplier": mult,
        "filename_DCC_coefficient_assumptions": dcc_coefs_file,
        "filename_households_resampled": household_data_file
    }

    # write JSON file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Created: {filepath}")

#%% DRY scenarios ###
# dry climate scenarios
real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]
dT_All = [5]
dP_All = [80]
dCV_All = [1.2]
demand_All = ['Baseline']

# policy: 1. desal, 2. dpr, 3. mcasr, 4. transfer soq, 5. transfer sv
decision_vars = [0.654, 0.4, 0.5, 0.3, 0.1, 0.2]
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/SA/DemandDemographics/'

# loop through all combinations of parameters and run the model
for pe, mf, dem, house, income in itertools.product(PriceElasticity, MultiFamily, SF_Demands_Multiplier, SF_Households, SF_Income):

    print('price elasticity: {}, multi-family demands: {}, single-family demand multiplier: {}, single-family # households: {}, income skew: {}'.format(pe, mf, dem, house, income))
    # define name to add to filenames
    name_add = f"PE={pe}_MF={mf}_Dem={dem}_House={house}_Income={income}_"

    # get multi-family demands
    demand_All = ['MF_{}'.format(mf)]
    print('demand all: {}'.format(demand_All))

    # define filepath SA name
    filepath_SA = f"../model_assumptions_and_scenarios/sensitivity_analysis_DemandDemographics/SA_PE={pe}_Dem={dem}_House={house}_Income={income}.json"

    # run model
    sim_model_run_v2(decision_vars, filepath, real_All, dT_All, dP_All, dCV_All, demand_All, filepath_SA, name_add)

#%% WET scenarios ###
# wet climate scenarios
real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]
dT_All = [0]
dP_All = [100]
dCV_All = [1.0]
demand_All = ['Baseline']

# policy: 1. desal, 2. dpr, 3. mcasr, 4. transfer soq, 5. transfer sv
decision_vars = [0.654, 0.4, 0.5, 0.3, 0.1, 0.2]
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/SA/DemandDemographics/'


for pe, mf, dem, house, income in itertools.product(PriceElasticity, MultiFamily, SF_Demands_Multiplier, SF_Households, SF_Income):

    print('price elasticity: {}, multi-family demands: {}, single-family demand multiplier: {}, single-family # households: {}, income skew: {}'.format(pe, mf, dem, house, income))
    # define name to add to filenames
    name_add = f"PE={pe}_MF={mf}_Dem={dem}_House={house}_Income={income}_"

    # get multi-family demands
    demand_All = ['MF_{}'.format(mf)]
    print('demand all: {}'.format(demand_All))

    # define filepath SA name
    filepath_SA = f"../model_assumptions_and_scenarios/sensitivity_analysis_DemandDemographics/SA_PE={pe}_Dem={dem}_House={house}_Income={income}.json"

    # run model
    sim_model_run_v2(decision_vars, filepath, real_All, dT_All, dP_All, dCV_All, demand_All, filepath_SA, name_add)


