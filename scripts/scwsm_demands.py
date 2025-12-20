"""
Note: This version expands a bunch of the classes from only working for desalination to working
for other infrastructure types (ASR and transfers)
"""

# IMPORT RELEVANT PACKAGES
import pywr.parameters
from pywr.parameters import Parameter
from pywr.parameters import load_parameter
import os
import numpy as np
import pandas as pd
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import math
import matplotlib.pyplot as plt
from household_demands import householdDemand
from Infrastructure_Financing_v1 import infraFinance
import random
import itertools
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# this class can be used to represent different infrastructure options and their matching characteristics across options
class Inf_Option:
    def __init__(self, name, capex, opex, deploy_time_yrs, inf_lifetime_yrs, order, inf_cap_mgd=None, inf_extr_mgd=None, note=None, order_cont=0):
        self.name = name
        self.capex = capex
        self.opex = opex
        self.deploy_time_yrs = deploy_time_yrs
        self.inf_lifetime_yrs = inf_lifetime_yrs
        self.order = order
        #self.order_cont = order_cont
        self.inf_cap_mgd = inf_cap_mgd # not using this currently
        self.inf_extr_mgd = inf_extr_mgd # not using this currently
        self.note = note

    def __repr__(self):
        return (f"Inf_Option(name={self.name}, capex={self.capex}, opex={self.opex}, deploy_time_yrs={self.deploy_time_yrs}, inf_lifetime_yrs={self.inf_lifetime_yrs},"
                f"order={self.order}, inf cap (mgd)={self.inf_cap_mgd}, inf extr (mgd)={self.inf_extr_mgd}, note={self.note})")

class HOUSEHOLD_DEMAND_DYNAMIC(Parameter):
    """
    This python class allows the urban demands to be updated dynamically at each timestep.
    The demand is composed of: (1) household demands using the DCC model, (2) all other demands (commercial, landscape, etc.)
    TO DO: Can use a constant value initially for (2) but this is probably not a good assumption

    The demands are constant daily throughout the month.

    """
    # Default filenames (class-level so they are easy to manage)
    DEFAULT_CASHFLOW_FILE = "cashflow_rate_assumptions.json"
    DEFAULT_COEFS_FILE = "DCCcoefs_v1.csv"
    DEFAULT_HHS_FILE = "resampled_income_data_30Nov2024.csv"

    def __init__(self, model,
                 parameters, filename_cashflow=None, filename_coefs=None, filename_hhs=None, **kwargs):  # add value function here possibly, also data_path for temp/hh data
        super().__init__(model)

        # add children parameters
        self.parameters = parameters
        for parameter in self.parameters:
            self.children.add(parameter)

        # --- Apply defaults if missing ---
        filename_cashflow = filename_cashflow or self.DEFAULT_CASHFLOW_FILE
        filename_coefs = filename_coefs or self.DEFAULT_COEFS_FILE
        filename_hhs = filename_hhs or self.DEFAULT_HHS_FILE
        print('i dont think we make it here')
        # --- Load household data ---
        filepath_hhs = f"../data/dcc_data/{filename_hhs}"
        filepath_coefs = f"../data/dcc_data/{filename_coefs}"

        # initialize instance of householdDemand class
        self.HHdemand = householdDemand()
        self.HHdemand.load_csv_hhdata(filepath_hhs)
        self.HHdemand.load_csv_dcc_coef(filepath_coefs)
        num_accts = len(self.HHdemand.HHdata)
        print('FILENAMES: cashflow: {}, coefs: {}, households: {}'.format(filename_cashflow, filename_coefs, num_accts))


        # --- Load cashflow JSON ---
        file_path = f"../model_assumptions_and_scenarios/{filename_cashflow}"
        with open(file_path) as f:
            self.params = json.load(f)

        for key, val in self.params.items():
            setattr(self, key, val)

        print('number of tiers in HOUSEHOLD_DEMAND_DYNAMIC: {}'.format(self.num_tiers))

        # hard code curtailment policy for now for dcc model
        self.curtail = 0

        # initialize numpy array to hold demand by tier
        num_months = len(pd.date_range(start=self.model.timestepper.start, end=self.model.timestepper.end, freq='MS'))
        self.arr_demand_by_tier = np.zeros((num_months, self.num_tiers))
        self.month = 0

        # get a random number for the mapped income
        self.hh_income_map = 1 # random.randint(1, 10)
        print('household income map: {}'.format(self.hh_income_map))

        # create 3d array of zeros for all household demands, bills, and ARs
        self.arr_hh_data = np.zeros(shape=(num_accts, num_months, 3))

        # create empty dataframe for holding household monthly demand data?
        self.df_demand_data = pd.DataFrame()

    def value(self, timestep, scenario_index):

        # update demand at the beginning of each month
        if timestep.day == 1:
            # try keeping volumetric prices constant

            date = pd.to_datetime(f"{timestep.year}-{timestep.month:02d}-{timestep.day:02d}")
            print(date)
            # we read in the monthly parameter values at the given timestep
            precipMM = self.model.parameters['precipitation_monthly_mm'].get_value(scenario_index)
            AET = self.model.parameters['evaporation_monthly_mm'].get_value(scenario_index)
            tempC = self.model.parameters['temperature_monthly_degC'].get_value(scenario_index)

            self.HHdemand.update_hh_data(self.model.parameters['water_rate_structure'].df_volPrices,
                                            self.model.parameters['water_rate_structure'].df_fixedFees, timestep.month,
                                            timestep.year, self.curtail, tempC, precipMM, AET, self.hh_income_map, self.model.parameters['factor_demand_multiplier'].get_value(scenario_index))

            # calculate the total demand across Santa Cruz
            Q_mgd = self.HHdemand.calcQ()

            if self.df_demand_data.empty:
                self.df_demand_data['demand'] = self.HHdemand.HHdata['demand']
            else:
                self.df_demand_data['demand'] += self.HHdemand.HHdata['demand']

            # get the total urban demand by tier
            self.arr_demand_by_tier[self.month, :] = self.HHdemand.sum_demand_by_tier()

            # get household data for all households
            self.arr_hh_data[:,self.month,:] = self.HHdemand.get_arr_hh_data()
            self.month += 1

            # obtain MF demand based on month of year
            mf_demand_mgd = self.model.parameters[
                'santa_cruz_MF_demand_MGD'].get_value(scenario_index)

            # obtain other demand based on the month of the year
            other_demand_mgd = self.model.parameters[
                'santa_cruz_demand_other_MGD'].get_value(scenario_index)

            # obtain losses based on the month of the year (constant)
            losses_mgd = self.model.parameters[
                'santa_cruz_losses_MGD'].get_value(scenario_index)

            # total demand
            self.demand_dynamic = Q_mgd + mf_demand_mgd + other_demand_mgd + losses_mgd 

        return self.demand_dynamic

    def set_dynamic_demand(self, new_value):
        self.demand_dynamic = new_value

    @classmethod
    def load(cls, model, data):
        """
        Pywr calls this method to load the parameter from JSON.
        Use safe `.get()` so missing fields do not raise KeyErrors.
        """
        #param_name = data.pop("name", None)
        parameters = data.pop("parameters")

        # extract filenames safely
        filename_cashflow = data.pop("filename_cashflow", None)
        filename_coefs = data.pop("filename_coefs", None)
        filename_hhs = data.pop("filename_hhs", None)

        return cls(model,
                   #param_name=param_name,
                   parameters=parameters,
                   filename_cashflow=filename_cashflow,
                   filename_coefs=filename_coefs,
                   filename_hhs=filename_hhs,
                   **data)


########## CHECK_PLANNING_INF PARAMETER ##########
class CHECK_PLANNING_INF(Parameter):
    """
    This python class returns a boolean variable with a 1 if we want to build a desalination plant and a 0 if we do not
    # need to figure out exactly where this is called
    """
    
    def __init__(self, model, parameters, filename=None, **kwargs): 
        super().__init__(model, **kwargs)
        # get filename
        if filename is None:
            raise ValueError("filename must be provided for CHECK_PLANNING_INF.")
        self.filename = filename
        print('filename: ', filename)

        # add children parameters
        self.parameters = parameters
        for parameter in self.parameters:
            self.children.add(parameter)


        # import data from json file
        file_name = "../model_assumptions_and_scenarios/" + self.filename
        with open(file_name) as f:
            self.params = json.load(f)  # json parser, could look at numpy and pandas parsers
        for key in self.params:
            setattr(self, key, self.params[key]) 

        # store infrastructure options in class Inf_Option in a dictionary with the options
        inf_options_dict = {inf_option['name']: Inf_Option(**inf_option) for inf_option in self.inf_options}

        self.inf_options_dict = inf_options_dict
        self.operations_monthly_fraction = np.array(self.operations_monthly_fraction)
        self.plan_inf = 0
        self.inf_count = 0
        self.df_demand_cumulative = pd.DataFrame()
        self.inf_cap = 0
        self.time_tracker = pd.DataFrame()

        # import percentiles and max res storage to get bins for rof triggers
        self.bins_storage_percent = np.array(self.bins_storage_percent, dtype=float)

        # get a list of names of the inf options
        inf_options = [inf_option.name for inf_option in self.inf_options_dict.values()]

        # List to store all combinations
        self.inf_combinations_list = []

        len_options = 0
        for r in range(4):  # This loops through combination lengths 0 to 3
            option_combinations = itertools.combinations(inf_options, r)
            len_options += math.comb(len(inf_options), r)

            for option in option_combinations:
                # Add current combination to the list
                self.inf_combinations_list.append(option)

        # add list for rof values
        self.list_rof = []

    # resets the parameter for each simulation- we use this function to set the infrastructure order after initialization
    # but before the model runs
    def reset(self):
        #print('reset up inf planning parameter')

        names = ['TRANSFER_SOQUEL', 'TRANSFER_SCOTTS_VALLEY', 'MCASR', 'DESALT_4MGD', 'DPR']
        # get a list of the values for each inf option
        for name in names: 
            param_name = 'order_' + name
            value = int(self.model.parameters[param_name].value(None, None))
            self.inf_options_dict[name].order = value

    def value(self, timestep, scenario_index):

        self.plan_inf = 0 #-- do we want to set this parameter back to zero? yes!
        
        # add deficit and demand values to pandas series
        date = pd.to_datetime(f"{timestep.year}-{timestep.month:02d}-{timestep.day:02d}")
        new_row = pd.DataFrame([{'Date': date, 'deficit': self.model.parameters['deficit_santa_cruz_demand_MGD'].get_value(scenario_index), 'demand': self.model.parameters['previous_time_step_demand'].get_value(scenario_index)}])
        
        self.df_demand_cumulative = pd.concat([self.df_demand_cumulative, new_row], ignore_index=True)


        # set this to ROF for risk-of-failure threshold dynamic modeling approach
        if self.check_conditions_rof(timestep, scenario_index): 
            self.plan_inf = 1
            self.inf_count += 1

            # get infrastructure based on order
            self.current_inf_option = self.find_inf_by_order(self.inf_count)

            # set up time tracker
            self.setup_time_tracker(date)

        return self.plan_inf
        

    def fraction_unmet_demand(self, window_size_yrs):
        # set window size to days
        window_size_days = 365 * window_size_yrs

        # get pandas series for deficit and demands
        deficit_series = pd.Series(self.df_demand_cumulative['deficit'])
        demand_series = pd.Series(self.df_demand_cumulative['demand'])
        
        # get moving averages
        movavg_deficit = deficit_series.rolling(window=window_size_days).mean()
        movavg_demand = demand_series.rolling(window=window_size_days).mean()
        
        return (movavg_deficit.iloc[-1] / movavg_demand.iloc[-1])

    # function to check conditions for building desal using rof triggers
    def check_conditions_rof(self, timestep, scenario_index):

        # get storage bin levels in first timestep
        # couldn't figure out how to get this in initialization because max_volume was zero, so this is a workaround
        if timestep.day == 1 and timestep.month == 10 and timestep.year == 2020:
            # get storage bin levels
            max_storage = self.model.nodes['ll_reservoir'].max_volume
            self.bins_storage_MG = self.bins_storage_percent / 100 * max_storage

        # only check conditions at beginning of a water year- Oct 1; also run the model for at least 2 years before checking
        if timestep.day == 1 and timestep.month == 10 and timestep.year >= 2021:

            date = pd.to_datetime(f"{timestep.year}-{timestep.month:02d}-{timestep.day:02d}")

            # 1. compile the required data
            # a. get current reservoir storage level
            LL_volume = self.model.nodes['ll_reservoir'].volume[0]

            # get the index of the storage value that is closest
            closest_storage_value = self.bins_storage_MG[np.abs(self.bins_storage_MG - LL_volume).argmin()]

            # b. get previous year's total demands
            arr = self.model.parameters['previous_time_step_demand'].arr_annual_demand_MG
            self.annual_demand_MG = arr[arr > 0][-1]

            # get the index of the demand value that is closest
            index_demand_bin = np.abs(self.bins_demand_MG - self.annual_demand_MG).argmin()

            # c. get planned and built infrastructure options (don't include options that have already ramped down)
            df_time_tracker = self.model.parameters['inf_time_tracker'].df_time_tracker

            # create a set of currently implemented inf options (or empty if no inf options implemented yet)
            # note: order does not matter for a set, but it does for a tuple
            # check if the df is empty first
            if df_time_tracker.empty:
                implemented_inf_options = set() # empty set
            else:
                # check if all infrastructure options have been implemented
                if len(df_time_tracker) >= 4: # fix this
                    print('no more inf options are left')
                    return 0
                # get all inf options that have some amount of lifetime remaining (deployed or being built)
                implemented_inf_options = set(df_time_tracker['inf_option'][df_time_tracker['inf_lifetime_month_remain'] > 0])

            # Find the index of the tuple with the same items, ignoring order
            index_inf = None
            for i, tup in enumerate(self.inf_combinations_list):
                if set(tup) == implemented_inf_options:
                    index_inf = i
                    break

            # 2. Import or call the right lookup table
            arr_rof_table = np.load('../data/rof_data/rof_table_st{}.npy'.format(round(closest_storage_value)))

            # 3. Get the current risk-of-failure trigger value (from lookup table)
            rof = arr_rof_table[index_demand_bin, index_inf]

            # 4. Save ROF value in list
            self.list_rof.append((timestep.year, rof))

            print('ll volume: {} MG (closest value: {}), annual demands: {} MG, inf options {}, '
                 'rof threshold: {}, rof value: {}'.format(round(LL_volume), round(closest_storage_value), round(self.annual_demand_MG), implemented_inf_options,
                                            self.model.parameters['threshold_rof'].get_value(scenario_index), rof))


            # 4. Check if current risk-of-failure trigger value exceeds threshold
            if rof >= self.model.parameters['threshold_rof'].get_value(scenario_index):
                return 1
            else:
                return 0

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)


    def find_inf_by_order(self, order_value):
        # Create a dictionary to map order values to User instances
        order_to_inf_dict = {inf_option.order: inf_option for inf_option in self.inf_options_dict.values()}
        return order_to_inf_dict.get(order_value, None)

    # helper function to get and process demand data
    def get_demand_data(self, timestep, scenario_index):
        # import the array with demand by tier data and sum it up
        #print('getting demand data')
        days_per_mo = np.tile(self.no_days_per_mo, self.yrs_to_run)

        ccf_to_mg = 748.052 / 1e6
        self.arr_demand = np.sum(self.model.parameters['previous_time_step_demand'].arr_demand_by_tier, axis=1)[0:self.yrs_to_run * 12] * ccf_to_mg
        self.arr_demand = self.arr_demand / days_per_mo
        return pd.DataFrame(self.arr_demand, columns=['Demand_MG'])
        

    def setup_time_tracker(self, date):
        months = 12
        
        # get dates
        plan_date = date
        deploy_date = plan_date + relativedelta(years=self.current_inf_option.deploy_time_yrs)
        payback_date = plan_date + relativedelta(years=self.payback_period_yrs)
        ramp_down_date = plan_date + relativedelta(years=(self.current_inf_option.deploy_time_yrs +
                                                          self.current_inf_option.inf_lifetime_yrs))

        self.time_tracker = pd.DataFrame([{'inf_option': self.current_inf_option.name,'desal_mgd': self.inf_cap, 'plan_date': plan_date, 'deploy_date': deploy_date,
                                           'payback_date': payback_date, 'ramp_down_date': ramp_down_date,
                        'deploy_month_remain': months*self.current_inf_option.deploy_time_yrs, 'payback_month_remain':
                                 months*self.payback_period_yrs, 'inf_lifetime_month_remain':
                                               months *(self.current_inf_option.deploy_time_yrs+self.current_inf_option.inf_lifetime_yrs)}])
        return self.time_tracker
        
    
    @classmethod
    def load(cls, model, data):
        print("DEBUG - incoming data:", data)  # ⬅ Add this
        filename = data.pop("filename")  # extract required arg
        parameters = data.pop("parameters")

        if filename is None:
            raise ValueError("filename must be provided for CHECK_PLANNING_INF.")
        if parameters is None:
            raise ValueError("parameters must be provided for CHECK_PLANNING_INF.")
        return cls(model, parameters=parameters, filename=filename, **data)


########## TIME_TRACKER PARAMETER ##########
class TIME_TRACKER(Parameter):
    """
    This python class tracks the deployment, payback period, and inf lifetime when desal is planned
    """
    
    def __init__(self, model, param_name, parameters, **kwargs):
        super().__init__(model, **kwargs)
        
        # add children parameters
        self.param_name = param_name
        self.parameters = parameters
        for parameter in self.parameters:
            self.children.add(parameter)
        
        self.time_tracker = 0
        self.df_time_tracker = pd.DataFrame()
        self.cols_to_modify = ['deploy_month_remain', 'payback_month_remain', 'inf_lifetime_month_remain']
        self.index_deploy = -1
        self.index_inf_lifetime = -1
        self.count_inf = 0
        self.df_inf_deploy = pd.Series(dtype='string')
        self.df_inf_rampdown = pd.Series(dtype='string')

    
    def value(self, timestep, scenario_index):

        # if the time tracker df is not empty and it's the first day of the month and we already have inf in the time tracker df
        if (not self.df_time_tracker.empty) and timestep.day == 1 and self.time_tracker == 1:
            self.df_time_tracker[self.cols_to_modify] = self.df_time_tracker[self.cols_to_modify] - 1
            
            # if we deploy a project
            col_name = 'deploy_month_remain'
            self.df_inf_deploy = self.get_inf_options_updating(col_name)

            # if we ramp down a project
            col_name = 'inf_lifetime_month_remain'
            self.df_inf_rampdown = self.get_inf_options_updating(col_name)

            if self.df_time_tracker['inf_lifetime_month_remain'].max() == -1:
                self.time_tracker = 0 # set back to zero when we don't need to track the time anymore
            
        # if we plan inf, setup time tracker/add new row
        if self.model.parameters['planning_inf'].get_value(scenario_index) == 1:

            self.df_time_tracker = pd.concat([self.df_time_tracker, self.model.parameters['planning_inf'].time_tracker], ignore_index=True)
            print('time tracker dataframe in time tracker parameter with new row: ', self.df_time_tracker)

            # count how many inf options have been planned
            self.count_inf += 1
            
            # keep track of if we need to track time b/c of desal building
            self.time_tracker = 1
            
        return self.time_tracker

    def get_inf_options_updating(self, col_name):
        if (self.df_time_tracker[col_name] == 0).any():
            # get the infrastructure option(s) we are deploying
            df_inf = self.df_time_tracker[self.df_time_tracker[col_name] == 0]['inf_option']
            return df_inf
        else:
            return pd.Series(dtype='string')

    # get the next index where deployment or decommissioning will take place
    def update_index(self, col_name):
        idx = 999
        # filter df to only include rows where deployment hasn't happened
        df_filtered = self.df_time_tracker[self.df_time_tracker[col_name] > 0]
        if not df_filtered.empty:
            idx = df_filtered[col_name].idxmin()  # find index of row with min value
        return idx

    def get_inf_options_triggered(self, col_name):
        df_inf = self.df_time_tracker[self.df_time_tracker[col_name] == 0]['inf_option']

    def get_months_to_deployment(self, scenario_index):
        if self.get_value(scenario_index) == 1:
            #print(self.df_time_tracker['deploy_month_remain'].iloc[self.index_deploy])
            return self.df_time_tracker['deploy_month_remain'].iloc[self.index_deploy]
        else:
            return -999
    
    def get_months_to_inf_end_of_life(self, scenario_index):
        if self.get_value(scenario_index) == 1:
            return self.df_time_tracker['inf_lifetime_month_remain'].iloc[self.index_inf_lifetime]
        else:
            return -999

    def is_desal_at_end_of_life(self, scenario_index):
        turn_plant_off = 0
        df = self.model.parameters['inf_time_tracker'].df_time_tracker
        filtered_df = df[df['inf_option'].str.startswith('DESALT')].copy()
        if (filtered_df['deploy_month_remain'] <= 0).any():
            filtered_df.loc[:,'is_active'] = (filtered_df['deploy_month_remain'] <= 0) & (filtered_df['inf_lifetime_month_remain'] > 0)
            #print(filtered_df)
            keep_plant_on = (filtered_df['is_active'].any() == 1)
            turn_plant_off = 1 - keep_plant_on
            #print('turn plant off: ', turn_plant_off)
        return turn_plant_off

    # this function is used in the UPDATE_DESALT_CAPACITY() parameter class to add desal capacity
    def add_desal_cap(self):
        df = self.model.parameters['inf_time_tracker'].df_time_tracker
        filtered_df = df[df['inf_option'].str.startswith('DESALT')].copy()
        cap_to_add = 0

        if (filtered_df['deploy_month_remain'] == 0).any():
            row_with_zero = filtered_df[filtered_df['deploy_month_remain'] == 0].copy()
            row_with_zero.loc[:, 'extracted_number'] = row_with_zero['inf_option'].str.extract(r'(\d+)MGD')
            row_with_zero.loc[:, 'extracted_number'] = pd.to_numeric(row_with_zero['extracted_number'], errors='coerce')
            cap_to_add = row_with_zero['extracted_number'].sum()
            #print('add desal cap: ', cap_to_add)
        return cap_to_add


    # this function is used in the UPDATE_DESALT_CAPACITY() parameter class to REMOVE desal capacity
    def remove_desal_cap(self):
        df = self.model.parameters['inf_time_tracker'].df_time_tracker
        filtered_df = df[df['inf_option'].str.startswith('DESALT')].copy()
        cap_to_remove = 0
        if (filtered_df['inf_lifetime_month_remain'] == 0).any():
            row_with_zero = filtered_df[filtered_df['inf_lifetime_month_remain'] == 0].copy()
            row_with_zero.loc[:, 'extracted_number'] = row_with_zero['inf_option'].str.extract(r'(\d+)MGD')
            row_with_zero.loc[:, 'extracted_number'] = pd.to_numeric(row_with_zero['extracted_number'], errors='coerce')
            cap_to_remove = row_with_zero['extracted_number'].sum()
            #print('remove desal cap: ', cap_to_remove)
        return cap_to_remove

    def get_inf_option_lifetime(self, scenario_index):
        if self.get_value(scenario_index) == 1:
            return self.df_time_tracker['inf_option'].iloc[self.index_inf_lifetime]
        else:
            return -999

    def get_desal_deployment_capacity_mgd(self, scenario_index):
        if self.get_value(scenario_index) == 1:
            return self.df_time_tracker['desal_mgd'].iloc[self.index_deploy]
        else:
            return 0
    
    def get_desal_capacity_to_remove_mgd(self, scenario_index):
        if self.get_value(scenario_index) == 1:
            return self.df_time_tracker['desal_mgd'].iloc[self.index_inf_lifetime]
        else:
            return 0
        
    @classmethod
    def load(cls, model, data):
        return cls(model, **data)
    
    
########## GET_PREVIOUS_DEMAND PARAMETER ##########
class GET_PREVIOUS_DEMAND(Parameter):
    """
    This python class returns the santa cruz demand value from the previous timestep
    """
    
    def __init__(self, model):
        super().__init__(model)

        self.demand = 0
        
        # initialize numpy array to hold demand by tier
        num_months = len(pd.date_range(start=self.model.timestepper.start, end=self.model.timestepper.end, freq='MS'))
        num_years = len(pd.date_range(start=self.model.timestepper.start, end=self.model.timestepper.end, freq='YS'))
        self.num_tiers = self.model.parameters['cashflow_model'].num_tiers
        self.arr_demand_by_tier = np.zeros((num_months, self.num_tiers))

        # compile annual demands into an array
        self.arr_annual_demand_MG = np.zeros(num_years)
        self.current_yearly_demand = 0
        self.current_yr = 0

    
    def value(self, timestep, scenario_index):
        
        self.demand = self.model.parameters['santa_cruz_demand_MGD'].get_value(scenario_index)
        self.current_yearly_demand += self.demand
        
        # get updated array every month
        if timestep.day == 1:
            self.arr_demand_by_tier = self.model.parameters['santa_cruz_demand_MGD'].arr_demand_by_tier

        # sum daily demands to get annual values
        if timestep.day == 1 and timestep.month == 10 and timestep.year > self.model.timestepper.start.year:
            self.arr_annual_demand_MG[self.current_yr] = self.current_yearly_demand
            self.current_yr += 1
            self.current_yearly_demand = 0 # reset this annually
        
        return self.demand
        
    @classmethod
    def load(cls, model, data):
        return cls(model, **data)


########## CASHFLOW_MODEL PARAMETER ##########
class CASHFLOW_MODEL(Parameter):
    """
    This python class returns a cashflow model once the model triggers planning infrastructure capacity, i.e., the desalination plant. 
    """

    def __init__(self, model, parameters, filename=None, **kwargs):
        super().__init__(model)

        # get filename
        if filename is None:
            raise ValueError("filename must be provided for CASHFLOW_MODEL.")
        self.filename = filename
        print('filename: ', filename)
    
        # import assumptions from json file
        file_name = "../model_assumptions_and_scenarios/" + self.filename
        with open(file_name) as f:
            self.params = json.load(f)  # json parser, could look at numpy and pandas parsers
        for key in self.params:
            setattr(self, key, self.params[key])

        # import children- planning_inf
        # add children parameters
        self.parameters = parameters

        self.days_per_yr = 365
        self.rate_types = ['Quant', 'IRF']
        self.num_inf = 0
        
        # check that pay-go and debt proportions equal 1, are both >= 0, and exist
        if self.check_value(self.debt_service_fraction) and self.check_value(self.pay_go_fraction):
            if self.debt_service_fraction >= 0 and self.pay_go_fraction >= 0 and (self.debt_service_fraction + self.pay_go_fraction) == 1:
                print('debt and pay-go values are valid')
            else:
                #print('debt and pay-go values must be modified')
                self.pay_go = 1 - self.debt
                
        # initialize time tracker attribute in this class
        self.df_time_tracker = pd.DataFrame()
                
        # set up monthly cashflow table
        #print('setting up cashflow model dataframe')
        date_range = pd.date_range(start=self.model.timestepper.start, end=self.model.timestepper.end, freq='MS')
        self.num_months = len(date_range)
        tier_data = self.tier_boundaries + self.tiers_quant_dollar_per_ccf + self.tiers_irf_dollar_per_ccf + [self.fee_rate_stabilization_dollar_per_ccf] + [0, 0, 0, 0]
        data_repeated = np.tile(tier_data, (self.num_months, 1))

        # set up tier ratios dynamically
        self.tier_ratios_quant = np.zeros(self.num_tiers)
        self.tier_ratios_irf = np.zeros(self.num_tiers)
        for i in range(self.num_tiers):
            if i == 0:
                self.tier_ratios_quant[i] = 1
                self.tier_ratios_irf[i] = 1
            else:
                self.tier_ratios_quant[i] = getattr(self, f'T{i+1}_T1_quant_ratio')
                self.tier_ratios_irf[i] = getattr(self, f'T{i+1}_T1_IRF_ratio')

        print('tier ratios quant: {}'.format(self.tier_ratios_quant))
        print('tier ratios irf: {}'.format(self.tier_ratios_irf))

        # loop through to set up the dataframe based on the number of tiers
        col_names = []
        for prefix in ['UpperBound', 'Quant', 'IRF']:
            for i in range(1, self.num_tiers + 1):
                col_names.append(f"{prefix}_T{i}")
        col_names.extend(['Rate_Stab', 'Annuity_monthly_dollars', 'PayGo_monthly_dollars', 'IRF_revenue_needed', 'Opex_monthly_dollars'])
        df_tier_data = pd.DataFrame(data_repeated, columns=col_names)
        self.df_cashflow = pd.DataFrame({'Date': date_range})
        self.df_cashflow = pd.concat([self.df_cashflow, df_tier_data], axis=1)
        self.df_cashflow.set_index('Date', inplace=True) # set index to date column
        
        # add updated columns with 'upd' at the end to dataframe with initial rates
        for rate in self.rate_types:
            for t in range(1, self.num_tiers+1):
                orig_name = rate + '_T' + str(t)
                upd_name = orig_name + '_upd'
                
                # add column in cashflow model dataframe
                self.df_cashflow[upd_name] = self.df_cashflow[orig_name]

        # initialize capex/opex values
        self.capex = 0
        self.opex = 0
        self.count_inf_plants = 0
        self.month = -1

        # setup dataframes with original and updated water rates
        self.setup_water_rate_dataframes()

        # set up arrays with quant and irf increases
        self.arr_quant_inc = np.zeros((self.num_months, self.num_tiers))
        self.arr_irf_inc = np.zeros((self.num_months, self.num_tiers))

        self.payback_period_yrs = self.model.parameters['planning_inf'].payback_period_yrs

    def value(self, timestep, scenario_index):

        # only run this part if we trigger building infrastructure in the current timestep
        #if we trigger planning desal
        if timestep.day == 1:
            self.month += 1
            
        if self.model.parameters['planning_inf'].get_value(scenario_index) == 1:
            self.num_inf += 1
            # get dates
            date = pd.to_datetime(f"{timestep.year}-{timestep.month:02d}-{timestep.day:02d}")
            plan_date = date

            self.deploy_time_yrs = self.model.parameters['planning_inf'].current_inf_option.deploy_time_yrs
            self.inf_lifetime_yrs = self.model.parameters['planning_inf'].current_inf_option.inf_lifetime_yrs
            deploy_date = plan_date + relativedelta(years=self.deploy_time_yrs)
            payback_date = plan_date + relativedelta(years=self.payback_period_yrs)
            ramp_down_date = plan_date + relativedelta(years=(self.deploy_time_yrs + self.inf_lifetime_yrs))

            # obtain capex and opex costs
            self.capex = self.model.parameters['planning_inf'].current_inf_option.capex
            self.opex = self.model.parameters['planning_inf'].current_inf_option.opex
            
            # add to time tracker- output here
            self.add2TimeTracker()
            
            # calculate monthly annuity and pay-go
            self.calcAnnuity()
            self.calcPayGo()

            # calculate monthly opex cost
            self.calcMonthlyOpex()
            
            # update cashflow model table
            self.updateCashflowTable(plan_date, deploy_date, payback_date, ramp_down_date)

            # get demand estimates by tier
            self.calcAvgDemand_by_Tier()

            # calculate theoretical rate increases
            self.arr_quant_inc = self.calc_rate_increase_v2('quant')
            self.arr_irf_inc = self.calc_rate_increase_v2('IRF')

            # update volumetric rate structure with theoretical demands
            # try getting avg rates during infrastructure period
            end_date = min(ramp_down_date, pd.to_datetime("2070-09-30"))
            self.update_water_rate_dataframe(plan_date, end_date)

            # calculate updated demands with theoretical rates
            self.calc_demands_with_updated_rates(timestep, True)

            # recalculate rates
            self.arr_quant_inc = self.calc_rate_increase_v2('quant')
            self.arr_irf_inc = self.calc_rate_increase_v2('IRF')

            # update cashflow model and table
            self.update_cashflow_with_rate_increase(True)

            print('month used in updating cashflow with rate increase: ', self.month)
        
            # add to count of desal plants
            self.count_inf_plants += 1
            
            # for testing
            #self.time_step_sample = self.get_timestep(timestep)

        return self.count_inf_plants



    # calculates the loan needed based on the proportion of debt and the annual annuity based on the payback period and
    # interest rate
    def calcAnnuity(self):
        print('interest rate: {}'.format(self.interest_rate_loan))
        self.loan_needed= self.debt_service_fraction * self.capex
        one_plus_r_n = (1+self.interest_rate_loan)**self.payback_period_yrs
        self.annuity_monthly = (self.loan_needed * (self.interest_rate_loan * one_plus_r_n) / (one_plus_r_n-1))/12
        # update annuity based on fraction of costs met by residential sector
        self.annuity_monthly = self.annuity_monthly * self.fraction_res_sector
        print('calc monthly annuity: ', self.annuity_monthly)
        return self.annuity_monthly

    # calculates the pay-as-you-go annual payment for the number of years it takes to deploy the technology
    def calcPayGo(self):
        self.pay_go_total = self.pay_go_fraction * self.capex
        self.pay_go_per_month = (self.pay_go_total / self.deploy_time_yrs) / 12
        # update pay-go based on fraction of costs met by residential sector
        self.pay_go_per_month = self.pay_go_per_month * self.fraction_res_sector
        print('calc monthly pay-go: ', self.pay_go_per_month)
        return self.pay_go_per_month

    # calculate the capex cost using the plant capacity, lifetime, lcow, and capex/opex ratio
    def calcCapex(self):
        # add something in if we don't have all of this information??
        self.capex = self.desal_mgd * self.days_per_yr * self.lcow  * self.capex_opex_ratio * (1/self.crf)
        #print('in calcCapex function- capex cost is: ', self.capex)
        return self.capex
        
    # calculate the opex cost using the plant capacity, lifetime, lcow, and opex ratio
    def calcOpex(self):
        self.opex = self.desal_mgd * self.lcow * self.days_per_yr * (1-self.capex_opex_ratio)
        self.opex_monthly = self.opex / 12 # self.inf_lifetime_yrs
        # update opex based on fraction of costs met by residential sector
        self.opex_monthly = self.opex_monthly * self.fraction_res_sector
        #print('in calcOpex function- monthly opex cost is: ', self.opex_monthly)
        return self.opex_monthly

    # calculate the monthly opex cost, which depends on the type of infrastructure
    def calcMonthlyOpex(self):
        inf_option = self.model.parameters['planning_inf'].current_inf_option
        if inf_option.name == 'TRANSFER':
            self.opex_monthly = inf_option.opex * self.model.parameters['transfer_yield'].transfer_yield_normal_operation_mgd
        else:
            self.opex_monthly = inf_option.opex / 12
        print('Monthly opex: {}'.format(self.opex_monthly))


    # function to add the total capex and annual opex costs to the time tracker dataframe from the time tracker parameter class
    def add2TimeTracker(self):
        # get the last row from the time tracker df
        self.df_new_row = self.model.parameters['inf_time_tracker'].df_time_tracker.iloc[-1].copy().to_frame().T
        self.df_new_row = self.df_new_row.drop(['deploy_month_remain', 'payback_month_remain', 'inf_lifetime_month_remain'], axis=1)
        # add capex and opex costs
        self.df_new_row.loc[:,'capex_cost'] = self.capex
        self.df_new_row.loc[:,'opex_cost_annual'] = self.opex
        # add the new row to the dataframe
        self.df_time_tracker = pd.concat([self.df_time_tracker, self.df_new_row], axis=0)

    
    # update the cashflow dataframe with the annuity, pay-go, IRF revenue needed, and opex monthly data
    def updateCashflowTable(self, start_plan_date, deploy_date, payback_date, inf_lifetime_date):

        # update annuity column
        end_date = min(payback_date, self.df_cashflow.index[-1].to_pydatetime())
        self.df_cashflow.loc[start_plan_date:end_date, 'Annuity_monthly_dollars'] += self.annuity_monthly
        
        # update pay-go column
        end_date = min(deploy_date, self.df_cashflow.index[-1].to_pydatetime())
        self.df_cashflow.loc[start_plan_date:end_date, 'PayGo_monthly_dollars'] += self.pay_go_per_month
        
        # update IRF revenue column
        self.df_cashflow.loc[:, 'IRF_revenue_needed'] = self.df_cashflow['Annuity_monthly_dollars'] + self.df_cashflow['PayGo_monthly_dollars']
        
        # update opex revenue column
        end_date = min(inf_lifetime_date, self.df_cashflow.index[-1].to_pydatetime())
        self.df_cashflow.loc[deploy_date:end_date, 'Opex_monthly_dollars'] += self.opex_monthly
    

    # calculate the number of months between a start and end date
    def calcMonthsBetween(self, start_date, end_date):
        # Convert the input strings to datetime objects if they are not already
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    
        # Calculate the number of years and months between the two dates
        year_diff = end_date.year - start_date.year
        month_diff = end_date.month - start_date.month
    
        # Total months difference including both start and end months
        total_months = year_diff * 12 + month_diff  # do I want +1 to include both start and end month?
    
        return total_months
    

    # function to check that values exist
    def check_value(self, value):
        # Check if value exists (is not None)
        if value is not None:
            # Check if value is not NaN (Not a Number)
            if not (isinstance(value, float) and math.isnan(value)):
                return True
        return False
    

    # function to get the average (or some other statistic) of the demand by tier over a previous time period (e.g., 10 yrs)
    def calcAvgDemand_by_Tier(self):
        # get the nonzero (updated) array of previous demands
        ind = np.nonzero(self.model.parameters['previous_time_step_demand'].arr_demand_by_tier[:,0])
        
        # update the indices based on how many past yrs we want to use to estimate future demands- e.g., first 10 yrs
        self.ind = ind[0][0:self.yrs_to_estimate_demand*12] # convert from tuple to array with [0]
        print('indices used for estimating future demands: ', self.ind)
        
        # update the in
        self.arr_demand_by_tier_nonzero = self.model.parameters['previous_time_step_demand'].arr_demand_by_tier[ind,:] # self.model.parameters[
        self.avg_demand_by_tier = np.mean(self.arr_demand_by_tier_nonzero, axis=1)
        self.med_demand_by_tier = np.median(self.arr_demand_by_tier_nonzero, axis=1)
                                               
        print('average demand by tier: ', self.avg_demand_by_tier)
        return self.avg_demand_by_tier[0]


    # function to calculate the percent change in demands with updated water rates
    def calc_demands_with_updated_rates(self, timestep, TF_print):

        # get two instances of the household demand class (and calculate demands) with two different volumetric rates and everything else, the same
        # what temperature, precip, AET data to use here? can use initialization data for now-- test sensitivity of this later
        self.HHdata_orig = self.calcDemands_HH(timestep, self.df_volPrices_orig)
        self.HHdata_upd = self.calcDemands_HH(timestep, self.df_volPrices_upd)

        # merge the dataframes and find the avg percent decrease (for each tier)
        self.calc_perc_decrease_by_tier()

        # apply the percent decreases to the average demands by tier
        if TF_print:
            print('avg fraction of total demands with rate increase: {}'.format(self.arr_frac_of_total))
            print('avg demand by tier before multiplication: {}'.format(self.avg_demand_by_tier))
        self.avg_demand_by_tier = self.avg_demand_by_tier * self.arr_frac_of_total
        if TF_print:
            print('avg demand by tier after multiplication: {}'.format(self.avg_demand_by_tier))


    # function to calculate rate increases for 'quant' or 'IRF' charges
    def calc_rate_increase_v2(self, ratio_type):
        if ratio_type != 'quant' and ratio_type != 'IRF':
            # print('invalid ratio type input-- switch to quantity')
            ratio_type = 'quant'

        if ratio_type == 'quant':
            col_name = 'Opex_monthly_dollars'
            tier_ratios = self.tier_ratios_quant
        if ratio_type == 'IRF':
            col_name = 'IRF_revenue_needed'
            tier_ratios = self.tier_ratios_irf

        # get denominator
        total_use_denom = np.sum(tier_ratios * self.model.parameters['cashflow_model'].avg_demand_by_tier)
        print('total use denom in calc_rate_increase function in CASHFLOW_MODEL: {}'.format(total_use_denom))

        # get array of zeros
        inc_zeros = np.zeros((self.month+1, self.num_tiers))
        print('size of zeros (months that have already passed): ', inc_zeros.shape)

        # get portion of cashflow model to update
        df_cashflow_update = self.model.parameters['cashflow_model'].df_cashflow.iloc[self.month+1:][col_name]
        print('size of non-zeros (future months): ', df_cashflow_update.shape)


        # get tier 1 increase
        inc_t1 = df_cashflow_update / total_use_denom

        # put tier increases together into array
        inc_nonzeros = np.zeros((len(inc_t1), self.num_tiers))
        for i in range(self.num_tiers):
            inc_nonzeros[:, i] = inc_t1 * tier_ratios[i]

        # concatenate arrays
        inc_all = np.row_stack((inc_zeros, inc_nonzeros))
        print('size of all: ', inc_all.shape)
        print('updated rates for {}: {}'.format(ratio_type, inc_all[inc_all > 0]))

        return inc_all


    # function to update the cashflow df with the rate increase
    def update_cashflow_with_rate_increase(self, TF_add_to_df):

        # create a backup of the ORIGINAL baseline only once
        if not hasattr(self, 'df_cashflow_baseline'):
            # Keep an immutable copy of the original rates (before any infra)
            col_names = []
            for t in range(1, self.num_tiers + 1):
                col_names.append(f"Quant_T{t}")
            for t in range(1, self.num_tiers + 1):
                col_names.append(f"IRF_T{t}")
            # store baseline (these are the original columns without "_upd")
            self.df_cashflow_baseline = self.df_cashflow[col_names].copy()

        # create a backup copy of the df
        # backup the current _upd columns for plotting comparisons (if desired)
        col_upd = []
        for t in range(1, self.num_tiers + 1):
            col_upd.append(f"Quant_T{t}_upd")
        for t in range(1, self.num_tiers + 1):
            col_upd.append(f"IRF_T{t}_upd")
        self.df_cashflow_backup = self.df_cashflow[col_upd].copy()
        
        # Store the arrays in a list for get_array
        self.array_list = [self.arr_quant_inc, self.arr_irf_inc]
        # Create a dictionary to map names to list indices
        self.array_index_map = {
            'Quant_inc': 0,
            'IRF_inc': 1
        }

        # If this is the first infra event, initialize *_upd columns from baseline
        first_inf = (self.count_inf_plants == 0)
        for rate in self.rate_types:
            #print('rate: {}'.format(rate))
            for t in range(1, self.num_tiers + 1):
                #print('tier: {}'.format(t))
                orig_name = f"{rate}_T{t}"
                upd_name = orig_name + "_upd"

                rate_inc = self.get_array(rate + '_inc')[:, t - 1].flatten()

                if upd_name not in self.df_cashflow.columns:
                    self.df_cashflow[upd_name] = self.df_cashflow[orig_name].copy()
                    # accumulate
                # update the values starting the first non-zero value
                idx = np.argmax(rate_inc != 0)
                self.df_cashflow.loc[self.df_cashflow.index[idx:], upd_name] = self.df_cashflow.loc[self.df_cashflow.index[idx:], orig_name] + rate_inc[idx:] # self.df_cashflow[upd_name]

                # also optionally keep per-infra increment column for audit / debugging
                if TF_add_to_df:
                    col_inc_name = f"{upd_name}_inc_inf{self.num_inf}"
                    self.df_cashflow[col_inc_name] = rate_inc


    # Function to dynamically select array
    def get_array(self, array_name):
        """
        Returns the array based on the array_name using the array index map.

        Parameters:
        array_name (str): The name of the array to retrieve.

        Returns:
        np.ndarray: The selected array.
        """
        if array_name in self.array_index_map:
            index = self.array_index_map[array_name]
            return self.array_list[index]
        else:
            raise ValueError(f"Array '{array_name}' not found. Available options are: {list(self.array_index_map.keys())}")


    # Function to set up dataframes for original and updated water rates- run in __init__ function
    def setup_water_rate_dataframes(self):

        # set up dataframes
        col_names = ['tiers', 'cutoff', 'quant_price', 'irf_price', 'price']
        self.df_volPrices_orig = pd.DataFrame(np.zeros((self.num_tiers, len(col_names))), columns=col_names)
        self.df_volPrices_upd = pd.DataFrame(np.zeros((self.num_tiers, len(col_names))), columns=col_names)

        for t in np.arange(1, self.num_tiers + 1):
            # print('tier = ', t)

            # update tiers
            self.df_volPrices_orig.loc[t - 1, 'tiers'] = t
            self.df_volPrices_upd.loc[t - 1, 'tiers'] = t

            # update cutoff
            col_name = 'UpperBound_T' + str(t)
            self.df_volPrices_orig.loc[t - 1, 'cutoff'] = \
                self.df_cashflow[col_name].iloc[0]
            self.df_volPrices_upd.loc[t - 1, 'cutoff'] = \
                self.df_cashflow[col_name].iloc[0]

            # update quant price-
            col_name = 'Quant_T' + str(t)
            self.df_volPrices_orig.loc[t - 1, 'quant_price'] = \
                self.df_cashflow[col_name].iloc[0]
            self.df_volPrices_upd.loc[t - 1, 'quant_price'] = \
                self.df_cashflow[col_name].iloc[0]

            # update irf price
            col_name = 'IRF_T' + str(t)
            self.df_volPrices_orig.loc[t - 1, 'irf_price'] = \
                self.df_cashflow[col_name].iloc[0]
            self.df_volPrices_upd.loc[t - 1, 'irf_price'] = \
                self.df_cashflow[col_name].iloc[0]

        # update marginal prices
        self.df_volPrices_orig.loc[:, 'price'] = self.df_volPrices_orig['quant_price'] + self.df_volPrices_orig[
            'irf_price'] + self.fee_rate_stabilization_dollar_per_ccf
        self.df_volPrices_upd.loc[:, 'price'] = self.df_volPrices_upd['quant_price'] + self.df_volPrices_upd[
            'irf_price'] + self.fee_rate_stabilization_dollar_per_ccf

        print('original df: ', self.df_volPrices_orig)
        print('updated df: ', self.df_volPrices_upd)
        return 1

    # function to update the dataframe with new water rates
    def update_water_rate_dataframe(self, plan_date, end_date):

        for t in np.arange(1, self.num_tiers + 1):

            # update quant price-
            col_name = 'Quant_T' + str(t) #+ '_upd'
            self.df_volPrices_upd.loc[t - 1, 'quant_price'] = self.df_cashflow.loc[plan_date:end_date, col_name].mean() + \
                max(self.arr_quant_inc[:, t - 1])

            # update irf price
            col_name = 'IRF_T' + str(t) #+ '_upd'
            self.df_volPrices_upd.loc[t - 1, 'irf_price'] = self.df_cashflow.loc[plan_date:end_date, col_name].mean() + \
                    max(self.arr_irf_inc[:, t - 1])
            #self.df_volPrices_upd.loc[t - 1, 'irf_price'] = self.df_volPrices_upd.loc[t - 1, 'irf_price'] + \
            #    np.max(self.arr_irf_inc[:, t - 1])

        # update marginal prices
        self.df_volPrices_upd.loc[:, 'price'] = self.df_volPrices_upd['quant_price'] + self.df_volPrices_upd[
            'irf_price'] + self.fee_rate_stabilization_dollar_per_ccf

        print('updated df: ', self.df_volPrices_upd)
        return 1
    
    # Function to get the timestep for teting of calcDemands_HH() function
    def get_timestep(self, timestep):
        return timestep
    
    # Function to calculate and output a dataframe with the household demands
    # can use this function for both sets of rates
    def calcDemands_HH(self, timestep, df_vol):
        self.curtail = 0 # switch this so not hard coded
        HHdemand = self.model.parameters['santa_cruz_demand_MGD'].HHdemand
        HHdemand.update_hh_data(df_vol, self.model.parameters['water_rate_structure'].df_fixedFees, timestep.month, timestep.year, self.curtail) # try not inputing AET, temp, precip and see if they initialize correctly
            
        # calculate the total demand across Santa Cruz
        HHdemand.calcQ()
        return HHdemand.HHdata
    
    # Function to calculate percent decreases in demands by tier
    def calc_perc_decrease_by_tier(self):
        
        #self.HHdata_orig
        col_names = ['account', 'demand', 'tier', 'marg_price']
        
        # merge HHdemand data dataframes-- future to do: put a check in here to make sure both dataframes exist
        subset_orig = self.HHdata_orig[col_names]
        subset_upd = self.HHdata_upd[col_names]
        merged_df = pd.merge(subset_orig, subset_upd, on='account', suffixes=('_orig', '_upd'))
        
        # calculate demand difference and percent difference
        merged_df['demand_diff'] = merged_df['demand_upd'] - merged_df['demand_orig']
        merged_df['demand_perc_change'] = merged_df['demand_diff'] / merged_df['demand_orig'] * 100

        # get average of demand percent change by original tier
        # future to do: test the sensitivity of using the avg vs another statistic
        self.arr_frac_of_total = np.zeros(self.num_tiers)
        for t in np.arange(1, self.num_tiers+1):
            #print('tier: ', t)
            criteria = (merged_df['tier_orig'] == t)
            filtered_df = merged_df[criteria]
            mean_demand = filtered_df['demand_diff'].mean()
            mean_perc = filtered_df['demand_perc_change'].mean()
            self.arr_frac_of_total[t-1] = mean_perc/100+1
            

    # function to plot the rates over time- can turn on/off       
    def plot_rates_over_time(self, boolean_TF, title):
        
        if boolean_TF:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            
            for t in np.arange(1, self.num_tiers+1):
                #print(t)
                # Quantity plot
                label_orig = 'Quant_T' + str(t)
                col_name_upd = label_orig + '_upd'
                axs[0].plot(self.df_cashflow_backup[col_name_upd], label=(label_orig+'_orig'))
                axs[0].plot(self.df_cashflow[col_name_upd], label=col_name_upd, linestyle='--')
                
                # IRF plot
                label_orig = 'IRF_T' + str(t)
                col_name_upd = label_orig + '_upd'
                axs[1].plot(self.df_cashflow_backup[col_name_upd], label=(label_orig+'_orig'))
                axs[1].plot(self.df_cashflow[col_name_upd], label=col_name_upd, linestyle='--')
            
            axs[0].legend()
            axs[0].set_title('Quantity vol charges comparison')
            axs[1].set_title('IRF vol charges comparison')
            fig.suptitle(title)
            plt.show()
    

    @classmethod
    def load(cls, model, data):
        print("DEBUG - incoming data:", data)  # ⬅ Add this
        filename = data.pop("filename")  # extract required arg
        parameters = data.pop("parameters")

        if filename is None:
            raise ValueError("filename must be provided for CASHFLOW_MODEL.")
        if parameters is None:
            raise ValueError("parameters must be provided for CASHFLOW_MODEL.")
        return cls(model, parameters=parameters, filename=filename, **data)


########## MONTH_TRACKER PARAMETER ##########
class MONTH_TRACKER(Parameter):
    """
    This is used in the ROF version of the model to track months. Not sure if this is actually being used in
    is_DESALT_included of type UPDATE_DESALT_ROF... check on this

    """

    def __init__(self, model):
        super().__init__(model)
        self.month_tracker = 0

    def value(self, timestep, scenario_index):
        if timestep.month == 1:
            self.month_tracker += 1

        return self.month_tracker

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)


########## WATER_RATE_STRUCTURE PARAMETER ##########
class WATER_RATE_STRUCTURE(Parameter):
    """
    This python class stores the different components of the water rate structure, including: (1) volumetric, (2) list of fixed fees by pipe size, (3) number of tiers, (4) rate stabilization fee, (5) other fixed fees (?), (6) surcharges.
    
    """

    def __init__(self, model, param_name, parameters):
        super().__init__(model)
        
        # add children parameters
        self.param_name = param_name
        self.parameters = parameters
        for parameter in self.parameters:
            self.children.add(parameter)
        
        # initialize custom parameter class attributes from cashflow_model parameter
        self.num_tiers = self.model.parameters['cashflow_model'].num_tiers
        self.tier_boundaries = self.model.parameters['cashflow_model'].tier_boundaries
        self.tiers_quant_dollar_per_ccf = self.model.parameters['cashflow_model'].tiers_quant_dollar_per_ccf
        self.tiers_irf_dollar_per_ccf = self.model.parameters['cashflow_model'].tiers_irf_dollar_per_ccf
        self.fee_rate_stabilization_dollar_per_ccf = self.model.parameters['cashflow_model'].fee_rate_stabilization_dollar_per_ccf
        self.surcharge = self.model.parameters['cashflow_model'].surcharge
        self.pipe_size = self.model.parameters['cashflow_model'].pipe_size
        self.fixed_rts_fees = self.model.parameters['cashflow_model'].fixed_rts_fees
        
        # update formatting/attributes
        fixedFees = {'pipe_size': self.pipe_size, 'fixed_rts_fees': self.fixed_rts_fees}
        self.df_fixedFees = pd.DataFrame(fixedFees)
        self.tiers = np.arange(1, self.num_tiers+1)

        # put together dataframe for volumetric prices
        volPrices = {'tiers': self.tiers, 'cutoff': self.tier_boundaries, 'quant_price': self.tiers_quant_dollar_per_ccf, 'irf_price': self.tiers_irf_dollar_per_ccf}
        self.df_volPrices = pd.DataFrame(volPrices)
        self.df_volPrices['price'] = self.df_volPrices['quant_price'] + self.df_volPrices['irf_price'] + self.fee_rate_stabilization_dollar_per_ccf
        
        self.month = 0


    def value(self, timestep, scenario_index):
        # update demand at the beginning of each month
        if timestep.day == 1:
            for t in np.arange(1, self.num_tiers+1):
    
                # update quant price
                col_name = 'Quant_T' + str(t) + '_upd'
                self.df_volPrices.loc[t-1, 'quant_price'] = self.model.parameters['cashflow_model'].df_cashflow[col_name].iloc[self.month]

                # update irf price
                col_name = 'IRF_T' + str(t) + '_upd'
                self.df_volPrices.loc[t-1, 'irf_price'] = self.model.parameters['cashflow_model'].df_cashflow[col_name].iloc[self.month]


            # update marginal prices
            self.df_volPrices.loc[:,'price'] = self.df_volPrices['quant_price'] + self.df_volPrices['irf_price'] + self.fee_rate_stabilization_dollar_per_ccf
            
            self.month += 1
        
        test = 5 # not needed
        return test
    
    @classmethod
    def load(cls, model, data):
        param_name = data['param_name']
        parameters = [load_parameter(model, parameter_data)
        for parameter_data in data.pop('parameters')]
            
        return cls(model, param_name, parameters)
