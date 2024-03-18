import os
from utils.utilities import src_dir

# BATTERY DEGRADATION
dod_degr = {
    'type': 'DOD',
    'battery_capex': 300_000,  # CAD/MWh
    'k_p': 1.14,  # Peukert lifetime constant, degradation parameter
    'N_fail_100': 6_000,  # number of cycles at DOD=1 until battery is useless
    'add_cal_age': False,  # adds fixed cost for calendar ageing if True via MAX-operator
    'battery_life': 20,  # expected battery life in years
}


# PLANT - ALBERTA - ENERGY ARBITRAGE - 2022
al4_bat_ea = {
    'env_name': 'al4_bat_ea',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'alberta3', 'alberta_2022_electricity_final.csv'),
    'state_vars': ['pool_price'],  # list of data columns to serve as state var
    # 'state_vars': ['pool_price', 'cos_h', 'sin_h'],  # list of data columns to serve as state var
    'grid': dict(
        demand_profile=None,
    ),
    'storage': dict(total_cap=10,  # MWh
                    max_soc=0.8,  # fraction of total capacity
                    min_soc=0.2,  # fraction of total capacity
                    max_charge_rate=2.5,  # MW
                    max_discharge_rate=2.5,  # MW
                    charge_eff=0.92,  # fraction
                    discharge_eff=0.92,  # fraction
                    aux_equip_eff=1.0,  # fraction, applied to charge & discharge
                    self_discharge=0.0,  # fraction, applied to every step (0 = no self-discharge)
                    init_strategy='half',  # 'min', 'max', 'half', or 'random'
                    degradation=dod_degr,
                    ),
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 8760,  # modeling period duration in hours
}
