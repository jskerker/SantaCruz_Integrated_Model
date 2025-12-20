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
sys.path.append('../../scripts/')
from Setup_SCWSM_Option_Analysis_CST import simSetup
print('import packages')

#%% Define climate and other scenarios to run ###

# Set up random climate simulations
def setup_random_sims(num_sims, random_seed):

    # set up climate scenarios
    random.seed(random_seed)
    real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]
    dT_All = np.arange(0, 6, 1)
    dP_All = np.arange(60, 121, 10)
    dCV_All = ['1.0', '1.1', '1.2']
    demand_All = ['Baseline', 'Low', 'High']

    # get random combinations of inputs
    combinations = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))
    random_combinations = random.sample(combinations, num_sims)

    return random_combinations

# set up climate simulations based on all combinations of inputs
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
#   arr_hh_data: household-level data
def sim_model_run_random(decision_vars, filepath, num_sims, random_seed, file_SA):

    # setup parameters for climate combinations
    # get climate combinations
    scenario = setup_random_sims(num_sims, random_seed)
    print(scenario)

    # get current datetime
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    # save climate scenario combinations
    headers = ['real', 'dT', 'dP', 'dCV', 'demand']
    # Save the list to a CSV file
    # with open(filepath + 'test_random_combinations_03Sept_NA.csv', 'w', newline='') as file:
    with open(filepath + 'climate_scenarios_{}_{}.csv'.format(num_sims, formatted_now), 'w', newline='') as file:
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
        modelSetup = simSetup(scenario[i][0], scenario[i][1], scenario[i][2], scenario[i][3], options, file_SA, scenario[i][4], decision_vars)
        model = modelSetup.m  # convert from simSetup class to pywr model object

        # run the model
        model.run()

        # post-processing
        # results dataframe
        df_results = post_process_results(model.to_dataframe())

        # cashflow df
        df_cashflow = model.parameters['cashflow_model'].df_cashflow
        # add total demands by tier to df_cashflow
        df_cashflow['demand_t1'] = model.parameters['previous_time_step_demand'].arr_demand_by_tier[:, 0]
        df_cashflow['demand_t2'] = model.parameters['previous_time_step_demand'].arr_demand_by_tier[:, 1]
        df_cashflow['demand_t3'] = model.parameters['previous_time_step_demand'].arr_demand_by_tier[:, 2]

        # df time tracker
        df_time_tracker = model.parameters['cashflow_model'].df_time_tracker

        # all household data
        arr_hh = model.parameters['santa_cruz_demand_MGD'].arr_hh_data

        # save results
        # dataframes to save- df_cashflow, df_rates, df_results,
        dataframes = [df_results, df_cashflow, df_time_tracker] # df_sample_low, df_sample_high, df_sample_random,
        filenames = ['df_results_P{}T{}_dCV{}_real{}_demand{}.csv'.format(scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4]),
                     'df_cashflow_P{}T{}_dCV{}_real{}_demand{}.csv'.format(scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4]),
                     'df_time_tracker_P{}T{}_dCV{}_real{}_demand{}.csv'.format(scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4])]

        # Iterate and save each DataFrame as a CSV
        for df, filename in zip(dataframes, filenames):
            df.to_csv(filepath + filename, index=True)

        filename = 'arr_hh_data_P{}T{}_dCV{}_real{}_demand{}.npy'.format(scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4])
        np.save(filepath+filename, arr_hh)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Model ran in {elapsed_time} seconds')

def sim_model_run_predetermined(decision_vars, filepath, real_All, dT_All, dP_All, dCV_All, demand_All, name_add, file_SA):

    # setup parameters for climate combinations
    # get climate combinations
    scenario = setup_climate_sims(real_All, dT_All, dP_All, dCV_All, demand_All)
    num_sims = len(scenario)
    print(scenario)

    # get current datetime
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    # save climate scenario combinations
    headers = ['real', 'dT', 'dP', 'dCV', 'demand']
    # Save the list to a CSV file
    with open(filepath + 'climate_scenarios_{}_{}.csv'.format(num_sims, formatted_now), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(scenario)

    # set up parameters for versions
    #version_parent = 'SCWSM-Option_Analysis'
    #version = 'SCWSM-SimOpt_Test'
    options = [None]

    # run one simulation at a time (I think I want to parallelize this)
    start_time = time.time()
    print('num sims: {}'.format(num_sims))
    for i in range(num_sims):
        print('simulation: {}'.format(str(i+1)))
        print(scenario[i])
        # create an instance of the model
        modelSetup = simSetup(scenario[i][0], scenario[i][1], scenario[i][2], scenario[i][3], options, file_SA, scenario[i][4], decision_vars)
        model = modelSetup.m  # convert from simSetup class to pywr model object

        # run the model
        model.run()

        # post-processing
        # results dataframe
        df_results = post_process_results(model.to_dataframe())

        # cashflow df
        df_cashflow = model.parameters['cashflow_model'].df_cashflow
        # add total demands by tier to df_cashflow
        df_cashflow['demand_t1'] = model.parameters['previous_time_step_demand'].arr_demand_by_tier[:, 0]
        df_cashflow['demand_t2'] = model.parameters['previous_time_step_demand'].arr_demand_by_tier[:, 1]
        df_cashflow['demand_t3'] = model.parameters['previous_time_step_demand'].arr_demand_by_tier[:, 2]

        # df time tracker
        df_time_tracker = model.parameters['cashflow_model'].df_time_tracker

        # all household data
        arr_hh = model.parameters['santa_cruz_demand_MGD'].arr_hh_data


        # save results
        # dataframes to save- df_cashflow, df_rates, df_results,
        dataframes = [df_results, df_cashflow, df_time_tracker] # df_sample_low, df_sample_high, df_sample_random,
        filenames = ['df_results_{}P{}T{}_dCV{}_real{}_demand{}.csv'.format(name_add, scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4]),
                     'df_cashflow_{}P{}T{}_dCV{}_real{}_demand{}.csv'.format(name_add,scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4]),
                     'df_time_tracker_{}P{}T{}_dCV{}_real{}_demand{}.csv'.format(name_add, scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4])]

        # Iterate and save each DataFrame as a CSV
        for df, filename in zip(dataframes, filenames):
            df.to_csv(filepath + filename, index=True)

        filename = 'arr_hh_data_{}P{}T{}_dCV{}_real{}_demand{}.npy'.format(name_add,scenario[i][2], scenario[i][1], scenario[i][3], scenario[i][0], scenario[i][4])
        np.save(filepath+filename, arr_hh)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Model ran in {elapsed_time} seconds')

#%% Baseline policy simulations ###
#Baseline Policy: ROF=0.654 1. desal, 2. dpr, 3. mcasr, 4. transfer soquel, 5. transfer sv
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/SA/Climate/' # for saving results
real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]
dT_All = [0, 1, 2, 3, 4, 5]
dP_All = [80, 90, 100, 110, 120]
dCV_All = [1.0, 1.1, 1.2] # 1.1, 1.2
demand_All = ['Baseline']
name_add = ''
file_SA = None
# Desal policy
decision_vars = [0.654, 0.4, 0.5, 0.3, 0.1, 0.2]
#sim_model_run_predetermined(decision_vars, filepath, real_All, dT_All, dP_All, dCV_All, demand_All, name_add, file_SA)


#%% No infrastructure simulations ###
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/' # for saving results
real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]
dT_All = [0, 1]
dP_All = [100]
dCV_All = [1.0]
demand_All = ['Baseline']
name_add = 'NoInf_'

decision_vars = [1.0, 0.3, 0.5, 0.4, 0.2, 0.1]
sim_model_run_predetermined(decision_vars, filepath, real_All, dT_All, dP_All, dCV_All, demand_All, name_add, file_SA)

#%% Random simulations ###
# Baseline Policy: ROF=0.654 1. desal, 2. dpr, 3. mcasr, 4. transfer soquel, 5. transfer sv
num_sims = 500
random_seed = 18
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure4/data_01Dec2025/'
decision_vars = [0.654, 0.4, 0.5, 0.3, 0.1, 0.2]
#sim_model_run_random(decision_vars, filepath, num_sims, random_seed, file_SA)

# High ROF Policy:
# ROF=0.89, 1. mcasr, 2. transfer sv, 3. dpr, 4. desal, 5. transfer soquel
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure5/HighROF_RandomSims/'
decision_vars = [0.89, 0.5, 0.2, 0.1, 0.4, 0.3] # updated this
sim_model_run_random(decision_vars, filepath, num_sims, random_seed, file_SA)

# Low ROF Policy:
# ROF=0.053, 1. desal, 2. transfer soquel, 3. dpr, 4. mcasr, 5. transfer sv
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure5/LowROF_RandomSims/'
decision_vars = [0.053, 0.2, 0.5, 0.4, 0.1, 0.3]
#sim_model_run_random(decision_vars, filepath, num_sims, random_seed, file_SA)


