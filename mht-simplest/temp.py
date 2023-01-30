"""
PredictEV Fleet

This script provides EV recommendations to electrify existing fleet yard.

@author: farhad.balali
version: 2021-10-1.0
"""
import math

import pandas as pd
from gekko import GEKKO
from typing import Dict, Tuple
from ypstruct import struct

from fetch_user_inputs import user_inputs_current_fleet_mix
from constants import  interest_rate, loan_duration, lease_residual_percentage, ev_category_map, average_electricity_cost, electrification_years, total_electrification_budget
from api.vehicles.models import ICEVehicle, EVVehicle
from api.vehicles_models.models import VehicleMaintenanceCost
from api.vehicles_models.utils.maps import MAINTENANCE_MAP, EV_CATEGORY_GROUP

import copy


class predictEV_fleet_composition:
    """
        This script is the collection of functions, utilities, conversion entities etc. related to the optimal
        calculation of the users' entered Internal Combustion Engine (ICE) vehicles fleet to corresponding Electric
        Vehicles (EV) fleet.

        ...

        Attributes
        ----------
        user_inputs : struct
            The formatted struct containing user's entered fleet vehicle in specific order

        Methods
        -------
        get_user_inputs_for_existing_fleet_mix()
            Prints the animals name and what sound it makes
        get_current_fleet_mix_info(existing_fleet_mix: Dict[str, dict])
        """

    def __init__(self, user_inputs: struct):
        """

        Parameters
        ----------
        user_inputs : struct
            The struct containing user's entered fleet vehicle in specific order maintained by backend.

        """
        self.user_inputs = user_inputs

    def get_user_inputs_for_existing_fleet_mix(self) -> dict[str, dict]:
        """
        Return users' fleet in ml compatible format.

        The backend formatted user fleet input received from backend is converted into ml compatible format and return as
        nested dictionary.

        Returns
        -------
        ml_compatible_user_inputs : dict[str, dict]
            The format of user fleet input is converted into ml compatible format and return as nested dictionaries
            with vehicle input types as keys and fleet details as values in dictionary.

        """
        ml_compatible_user_inputs = user_inputs_current_fleet_mix(self.user_inputs).user_input_information()
        return ml_compatible_user_inputs

    def get_current_fleet_mix_info(self, existing_fleet_mix: Dict[str, dict]):
        """
        As of now, this function has no functional role in the application.

        Parameters
        ----------
        existing_fleet_mix: Dict[str, dict]
            The ml compatible format of users' entered ICE fleet.

        Returns
        -------
        existing_fleet_mix: Dict[str, dict]
            The ml compatible format of users' entered ICE fleet.

        """

        for key, val in existing_fleet_mix.items():
            uuid = val['uuid']
            main_class = val['vehicle_category']
            make = val['make']
            model = val['model']

            try:
                # val['FHWA'] = ICEVehicle.objects.get(uuid=uuid).fhwa_class.class_group
                # val['FHWA'] = ICE_dataBase[(ICE_dataBase['Class'] == main_class) & (ICE_dataBase['Make'] == make) & (ICE_dataBase['Model'] == model)]['FHWA Class'].to_list()[0]
                existing_fleet_mix[key] = val

            except:
                print(main_class, make, model, 'Does Not Exist in ICE Database')

        return existing_fleet_mix

    def get_filter_EV_dataBase(self, **kwargs) -> pd.DataFrame:
        """

        Parameters
        ----------
        kwargs

        Returns
        -------
        EV_dataBase : pd.DataFrame
            The pandas dataframe containing corresponding EV vehicles and their details in columns for the users'
            entered ICE vehicles. Filtering criteria from database is matching parent category.

        """
        EV_queryset = EVVehicle.objects.filter(category__in=EV_CATEGORY_GROUP) \
            if kwargs['Vehice Class'] in EV_CATEGORY_GROUP \
            else EVVehicle.objects.filter(category=kwargs['Vehice Class']) \
            if kwargs['Vehice Class'] \
            else EVVehicle.objects.filter(parent_category=kwargs['Parent Category'])

        EV_queryset = EV_queryset.values('uuid', 'category', 'classification', 'parent_category', 'make', 'model',
                                         'year', 'fhwa_class', 'electric_range', 'mpge_city', 'mpge_highway',
                                         'battery_capacity', 'drive_load', 'max_rated_power', 'connector_type',
                                         'horsepower', 'drive_type', 'passenger_capacity', 'cargo_capacity_lbs',
                                         'msrp', 'nominal_battery_efficiency', 'battery_cycles', 'time_to_charge_hr',
                                         'weight')
        EV_dataBase = pd.DataFrame(list(EV_queryset))
        EV_dataBase.columns = ['uuid', 'Vehice Class', 'Classification', 'Parent Category', 'Make', 'Model', 'Year',
                               'FHWA Class', 'Electric Range', 'Fuel Economy Electric City (MPGe)',
                               'Fuel Economy Electric Hwy (MPGe)', 'Battery Capacity (kW)',
                               'Drive Load (kWh per 100 mile)', 'Maximum/rated power (kW)',
                               'Connector Type', 'Horsepower', 'RWD/AWD', 'Passenger capacity', 'Cargo capacity (lb)',
                               'MSRP ($, before incentives)', 'Nominal battery Efficiency',
                               'Number of battery cycles at this efficiency (sugg. lifetime of battery)',
                               'Time to Charge (hrs)', 'Weight (lbs)']

        category = kwargs['Vehice Class'] if kwargs['Vehice Class'] else kwargs['Parent Category']
        parent_category = MAINTENANCE_MAP.get(category, category)
        maintenance_per_mile = VehicleMaintenanceCost.objects \
            .filter(parent_category=parent_category, fuel_type='All-Electric Vehicle (EV)') \
            .first().cost_per_mile

        EV_dataBase['Est. Maintenance ($ per mile)'] = maintenance_per_mile
        EV_dataBase['Maximum/rated power (kW)'].fillna(400, inplace=True)
        drive_load_factor = pd.Series([3370.5] * len(EV_dataBase))

        average_mpge = []
        for idx, row in EV_dataBase.iterrows():
            if row['Vehice Class']:
                average_mpge.append(
                    (row['Fuel Economy Electric City (MPGe)'] + row['Fuel Economy Electric Hwy (MPGe)']) / 2)
            else:
                average_mpge.append(row['Fuel Economy Electric City (MPGe)'])

        EV_dataBase['Drive Load (kWh per 100 mile)'] = drive_load_factor / average_mpge

        return EV_dataBase

    def get_monthly_finance_cost(self, msrp: pd.Series) -> pd.Series:
        """
        Given the msrp cost of the EV vehicles, calculate the monthly finance cost for the duration of one year.

        Parameters
        ----------
        msrp: pd.Series
            The pandas' series having msrp cost of the vehicles in float.

        Returns
        -------
        monthly_payment: pd.Series
            Monthly payable finance cost for the given vehicles msrp price. It is negative implying users have to
            spend that amount monthly for the period of one year.
        """

        interest_rate_monthly = interest_rate / 12

        monthly_payment = msrp * (interest_rate_monthly * ((1 + interest_rate_monthly) ** (loan_duration * 12))) / (
                    1 - (1 + interest_rate_monthly) ** (loan_duration * 12))

        return -monthly_payment

    def get_monthly_lease_cost(self, msrp: pd.Series) -> pd.Series:
        """
        Given the msrp cost of the EV vehicles, calculate the monthly lease cost for the duration of one year.

        Parameters
        ----------
        msrp: pd.Series
            The pandas' series having msrp cost of the vehicles in float.

        Returns
        -------
        monthly_payment: pd.Series
            Monthly payable lease cost for the given vehicles msrp price. It is negative implying users have to
            spend that amount monthly for the period of one year.

        """
        interest_rate_monthly = interest_rate / 12

        monthly_payment = (msrp * lease_residual_percentage) * (
                    interest_rate_monthly * ((1 + interest_rate_monthly) ** (loan_duration * 12))) / (
                                      1 - (1 + interest_rate_monthly) ** (loan_duration * 12))

        return -monthly_payment

    def fleet_composition_optimization(self, existing_fleet_mix: Dict[str, dict]) -> Tuple[Dict[int, Dict[str, int]],
                                                                                           Dict[int, Dict[str, int]],
                                                                                           Dict[str, list],
                                                                                           Dict[int, Dict[str, int]],
                                                                                           Dict[str, list],
                                                                                           Dict[str, float], float,
                                                                                           Dict[str, int]]:
        """
        This is the method where actual EV fleet recommendation against ICE fleet is performed.

        First 'existing_fleet_mix' is used to initialize different variables and placeholders required for holding ICE
        fleet particulars. Then on the basis of ICE fleet parent categories, equivalent EVs are pulled from database in
        a for loop against each ICE vehicle. This is frst filtering step. In next filtering step, only those EVs are
        retained that fulfil the max_distance_travelled by the ICE vehicle on any day of the week. This is performed
        by using EVs battery capacity and drive load. The EVs successfully passing these two criteria are then fed into
        the optimization algorithm (mixed-integer programming (MIP) model) along with their monthly finance and lease
        costs, calculated separately. The model is also supplied with the required constraints that it must fulfill
        during optimization. Against each ICE vehicle, optimization model recommends only one EV that has the least
        finance/lease cost, and it fulfills the provided constraints. If second filtering step returns no EV, it
        implies required battery capacity (effectively means drive load) is exceedingly higher than any of the
        available EV in database. In this case "negative state of charge" henceforth called as negative soc appears if
        any oof the vehicle is suggested. For this case, negative soc is calculated as a warning dictionary pinpointing
        all the shifts that require higher battery capacity than available highest battery capacity EV.


        Parameters
        ----------
        existing_fleet_mix: Dict[str, dict]
            The ml compatible format of users' entered ICE fleet.

        Returns
        -------
        adjusted_electrification_rate: Dict[int, Dict[str, int]]

        final_output_lease_expanded_annual: Dict[int, Dict[str, int]]
            Dictionary holding the recommended EV against the ICE vechile with optimization criteria fulfilled on basis
            of the lease cost annually. This has no functional role in application as of now.

        final_output_lease_expanded_overall: Dict[str, list]

        final_output_finance_expanded_annual: Dict[int, Dict[str, int]]
            Dictionary holding the recommended EV against the ICE vechile with optimization criteria fulfilled on basis
            of the finance cost annually. This has no functional role in application as of now.

        final_output_finance_expanded_overall: Dict[str, list]

        electrified_vehicle_tracker: Dict[str, float]
            Dictionary holding the successful recommendation of the EV against ICE as per the quantity flag coming from
            user. This has no functional role in application as of now.

        total_electrification_budget_recomm_sol: float
            Total budget required for electrification of the ICE fleet to EV. This has no functional role in
            application as of now.

        warning_shifts: Dict[str, int]]
            Dictionary containing those shift details that required higher battery capacity than available in any EV.
            The input_category is key while violating indices are placed as value in the dict.


        """
        final_output_lease = {}
        final_output_finance = {}

        final_output_lease_expanded_annual = {}
        final_output_finance_expanded_annual = {}

        # Filter input categories based on input_category since electrification optimization is based on maximum
        # values of quantity in fleet for each inout_category defined by user.

        seen_input_category = []
        unique_input_category = {}

        for key, val in existing_fleet_mix.items():
            if val['input_category'] not in unique_input_category:
                unique_input_category[val['input_category']] = [val['quantity_in_fleet'], key]
                seen_input_category.append(val['input_category'])
            elif unique_input_category[val['input_category']][0] < val['quantity_in_fleet']:
                unique_input_category[val['input_category']] = [val['quantity_in_fleet'], key]

        existing_fleet_mix_modified = {}
        for key, val in unique_input_category.items():
            existing_fleet_mix_modified[val[1]] = existing_fleet_mix[val[1]]

        #remaining_electrification_budget = total_electrification_budget
        total_electrification_budget_recomm_sol = 0

        adjusted_electrification_rate = {}

        electrified_vehicle_tracker = {}
        warning_shifts = {}

        for key, val in existing_fleet_mix_modified.items():
            electrified_vehicle_tracker[key] = existing_fleet_mix[key]['quantity_in_fleet']

        for year in range(electrification_years):

            final_output_lease = {}
            final_output_finance = {}

            adjusted_electrification_rate[year] = {}
            remaining_electrification_budget = total_electrification_budget[year]

            for key, val in existing_fleet_mix_modified.items():
                elelctrification_rate = 1

                info = {'Vehice Class': existing_fleet_mix[key]['vehicle_category'],
                        'Parent Category': existing_fleet_mix[key]['parent_category']}
                #         'FHWA Class' :  3}
                EV_dataBase = self.get_filter_EV_dataBase(**info)

                # filtering out vehicles whose battery capacity can't fulfil the energy requirement
                drive_load = EV_dataBase['Drive Load (kWh per 100 mile)']
                required_energy = (drive_load * existing_fleet_mix[key]['max_shift_miles']) / 100
                required_battery_capacity = required_energy + (
                            required_energy * (1 - EV_dataBase['Nominal battery Efficiency']))
                # select only vehicles that fulfill min criteria
                EV_dataBase_filtered = EV_dataBase[EV_dataBase["Battery Capacity (kW)"] > required_battery_capacity]
                # Reset the index
                EV_dataBase_filtered.reset_index(drop=True, inplace=True)

                if EV_dataBase_filtered.empty:
                    EV_with_lower_ranges = EV_dataBase[EV_dataBase["Battery Capacity (kW)"] <= required_battery_capacity]
                    input_category = existing_fleet_mix[key]['input_category']
                    max_range = EV_with_lower_ranges['Electric Range'].max()
                    EV_with_max_range = EV_with_lower_ranges.loc[EV_with_lower_ranges['Electric Range'] == max_range]
                    battery_capacity = EV_with_max_range['Battery Capacity (kW)'].to_list()[0]
                    drive_load = EV_with_max_range['Drive Load (kWh per 100 mile)'].to_list()[0]
                    nominal_battery_efficiency = EV_with_max_range['Nominal battery Efficiency'].to_list()[0]
                    max_available_range = 100 * (
                                battery_capacity - battery_capacity * (1 - nominal_battery_efficiency)) / drive_load
                    max_available_range = math.floor(max_available_range)
                    input_keys = {idx: input_key for idx, (input_key, value) in enumerate(existing_fleet_mix.items())
                                  if value['input_category'] == input_category}
                    shift_miles = {idx: existing_fleet_mix[input_key]['max_range_per_day']
                                   for idx, input_key in input_keys.items()}
                    warning_shifts_indexes = [idx for idx, miles in shift_miles.items()
                                              if miles > max_available_range]
                    warning_shifts[input_category] = warning_shifts_indexes
                    continue

                # cost of drive load per month based on distance travel in each month
                driveLoad = EV_dataBase_filtered['Drive Load (kWh per 100 mile)']
                # total_driveLoad_cost_monthly = driveLoad * existing_fleet_mix[key]['quantity_in_fleet'] * existing_fleet_mix[key]['days_in_operation'] *  existing_fleet_mix[key]['hours_of_operation']
                # total monthly required energy (kWh)
                total_energy_cost_monthly = driveLoad * (existing_fleet_mix[key]['days_in_operation'] * 4) * (existing_fleet_mix[key]['average_range_per_day'] / 100) * average_electricity_cost

                financeCost = self.get_monthly_finance_cost(EV_dataBase_filtered['MSRP ($, before incentives)'])
                leaseCost = self.get_monthly_lease_cost(EV_dataBase_filtered['MSRP ($, before incentives)'])

                # fixed_maintenance = EV_dataBase_filtered['Est. Maintenance ($ per year)'] / 12
                variable_maintenance = EV_dataBase_filtered['Est. Maintenance ($ per mile)'] * (existing_fleet_mix[key]['days_in_operation'] * 4) * existing_fleet_mix[key]['average_range_per_day']
                total_maintenance = variable_maintenance

                TCFinance = (financeCost + total_maintenance + total_energy_cost_monthly).to_list()
                TCLease = (leaseCost + total_maintenance + total_energy_cost_monthly).to_list()

                mdl = GEKKO(remote=False)
                num_EV_lease = mdl.Array(mdl.Var,len(EV_dataBase_filtered),lb=0,ub=None,integer=True)
                num_EV_finance = mdl.Array(mdl.Var,len(EV_dataBase_filtered),lb=0,ub=None,integer=True)


                mdl.Minimize(self.objective_function(num_EV_lease, num_EV_finance, TCLease, TCFinance))
                mdl.Equation(sum(num_EV_lease) + sum(num_EV_finance) == electrified_vehicle_tracker[key] * elelctrification_rate)
                # mdl.Equation((sum(num_EV_lease * leaseCost) + sum(num_EV_finance * financeCost)) <= (remaining_electrification_budget/12))

                mdl.options.SOLVER = 1
                mdl.solve()

                # remaining electrification budget get updated after optimizing for each input_category
                for i in range(len(num_EV_lease)):
                    remaining_electrification_budget -= (num_EV_lease[i][0] * leaseCost[i] * 12) + (num_EV_finance[i][0] * financeCost[i] * 12)

                adjusted_electrification_rate[year][key] = elelctrification_rate

                for i in range(len(num_EV_lease[0])):
                    # monthly cost
                    total_electrification_budget_recomm_sol += num_EV_lease[0][i] * leaseCost[i]
                    electrified_vehicle_tracker[key] -= num_EV_lease[0][i]

                for i in range(len(num_EV_finance[0])):
                    # monthly cost
                    total_electrification_budget_recomm_sol += num_EV_finance[0][i] * financeCost[i]
                    electrified_vehicle_tracker[key] -= num_EV_finance[0][i]

                output_lease = []
                output_finance= []

                for i in range(len(EV_dataBase_filtered)):
                    if num_EV_lease[i][0] > 0:
                         selected_EV_info = EV_dataBase_filtered.iloc[i].to_frame().T
                         selected_EV_info['Optimal Value'] = int(num_EV_lease[i][0])
                         output_lease.append(selected_EV_info)

                    if num_EV_finance[i][0] > 0:
                        selected_EV_info = EV_dataBase_filtered.iloc[i].to_frame().T
                        selected_EV_info['Optimal Value'] = int(num_EV_finance[i][0])
                        output_finance.append(selected_EV_info)

                if len(output_lease) > 0:
                    final_output_lease[key] = output_lease

                if len(output_finance) > 0:
                    final_output_finance[key] = output_finance


                final_output_lease_expanded_annual[year] = {}
                final_output_finance_expanded_annual[year] = {}

                map = {}

                for key, val in unique_input_category.items():
                    map[val[1]] = []

                for key in map.keys():
                    for k, v in existing_fleet_mix.items():
                        if existing_fleet_mix[key]['input_category'] == existing_fleet_mix[k]['input_category']:
                            map[key].append(k)


                for key in map.keys():
                    for val in map[key]:
                        try:
                            final_output_lease_expanded_annual[year][val] = copy.deepcopy(final_output_lease[key])
                            final_output_lease_expanded_annual[year][val][0]['Optimal Value'] = min(final_output_lease[key][0]['Optimal Value'].to_list()[0], existing_fleet_mix[val]['quantity_in_fleet'])


                            final_output_finance_expanded_annual[year][val] = copy.deepcopy(final_output_finance[key])
                            final_output_finance_expanded_annual[year][val][0]['Optimal Value'] = min(final_output_finance[key][0]['Optimal Value'].to_list()[0],existing_fleet_mix[val]['quantity_in_fleet'])

                        except:
                            pass

            final_output_lease_expanded_overall = {}
            final_output_finance_expanded_overall = {}

            for val in final_output_lease_expanded_annual.values():
                for k, v in val.items():
                    if k not in final_output_lease_expanded_overall:
                        final_output_lease_expanded_overall[k] = v.copy()
                    else:
                        final_output_lease_expanded_overall[k][0]['Optimal Value'] += v[0]['Optimal Value']

            for val in final_output_finance_expanded_annual.values():
                for k, v in val.items():
                    if k not in final_output_finance_expanded_overall:
                        final_output_finance_expanded_overall[k] = v.copy()
                    else:
                        final_output_finance_expanded_overall[k][0]['Optimal Value'] += v[0]['Optimal Value']

        return adjusted_electrification_rate, final_output_lease_expanded_annual, final_output_lease_expanded_overall, \
               final_output_finance_expanded_annual, final_output_finance_expanded_overall, \
               electrified_vehicle_tracker, total_electrification_budget_recomm_sol, warning_shifts

    def objective_function(self, x, y, TC_Lease, TC_Finance):
        """

        Parameters
        ----------
        x
        y
        TC_Lease
        TC_Finance

        Returns
        -------

        """
        sumT = 0
        for i in range(len(x)):
            sumT += (TC_Lease[i] * x[i]) + (TC_Finance[i] * y[i])

        return sumT

    def get_ev_database(self) -> pd.DataFrame:
        """
        As of now, this function has no functional role in the application.

        Returns
        -------
        ev_database: pd.DataFrame
            The EV dataframe read from the Excel sheet.

        """

        ev_database = pd.read_excel('../data/Fleet_DB_Example.xlsx', sheet_name='EV')

        return ev_database

    def get_ice_database(self) -> pd.DataFrame:
        """
        As of now, this function has no functional role in the application.

        Returns
        -------
        ice_database: pd.DataFrame

        """
        ice_database = pd.DataFrame(list(ICEVehicle.objects.all().values(
            'category', 'make', 'model', 'year', 'fhwa_class', 'horsepower', 'drive_type',
            'passenger_capacity', 'cargo_capacity_lbs', 'mpg_city', 'mpg_highway', 'msrp'
        )))

        ice_database.columns = ['Class', 'Make', 'Model', 'Year', 'FHWA Class', 'Horsepower',
                                'Drive Type(RWD / AWD / FWD', 'Passenger capacity', 'Cargo capacity(lb)',
                                'MPG - city', 'MPG - highway', 'MSRP($)']

        return ice_database
