'''in this script, we consider constraints corresponding to the current multi battery environment,
that is to say the voltage regulation only considers that constrainting the limitation connected to the nodes connected to batteries,
Using 2020 11-6 as the example'''
import numpy as np
from environments.env import PowerNetEnv
from pyomo.environ import *
import pandas as pd


def construct_opf_model(Vnom, Vmin, Vmax, Data_Network):
    # Data Processing
    battery_parameters = {
        'capacity': 1.0,  # MW.h
        'max_charge': 0.3,  # MW
        'max_discharge': 0.3,  # MW
        'efficiency': 1,
        'degradation': 0,  # euro/kw
        'max_soc': 0.8,
        'min_soc': 0.2,
        'initial_soc': 0.4}
    TIMES=Data_Network['TIMES']
    NODES = Data_Network['NODES']
    LINES = Data_Network['LINES']
    Tb = Data_Network['Tb']
    PD = Data_Network['PD']
    QD = Data_Network['QD']
    R = Data_Network['R']
    X = Data_Network['X']
    BATTERY_NODES = Data_Network['BATTERY_NODES']
    # Type of Model
    model = ConcreteModel()

    # Define Sets
    model.NODES = Set(initialize=NODES)
    model.LINES = Set(initialize=LINES)
    model.TIMES = Set(initialize=TIMES)

    # Define Parameters
    model.Vnom = Param(initialize=Vnom, mutable=False)
    model.Vmin = Param(initialize=Vmin, mutable=False)
    model.Vmax = Param(initialize=Vmax, mutable=False)
    model.Tb = Param(model.NODES, initialize=Tb, mutable=True)
    # model.PD = Param(model.TIMES,model.NODES, initialize=0, mutable=True)  # Node demand
    model.QD = Param(model.TIMES,model.NODES, initialize=0, mutable=True)  # Node demand
    model.R = Param(model.LINES, initialize=R, mutable=False)  # Line resistance
    model.X = Param(model.LINES, initialize=X, mutable=False)  # Line resistance
    ## define parameters for battery
    model.battery_initial_soc=Param(default=battery_parameters['initial_soc'])
    model.battery_capacity=Param(default=battery_parameters['capacity'])
    model.battery_soc_max=Param(default=battery_parameters['max_soc'])
    model.battery_soc_min=Param(default=battery_parameters['min_soc'])
    model.battery_max_change=Param(default=battery_parameters['max_charge'])






    # define initialize PD
    def PD_init_rule(model,time,node):
        model.PD[time,node]=PD[time,node]
        return (model.PD[time,node])
    model.PD=Param(model.TIMES,model.NODES,initialize=PD_init_rule)

    def R_init_rule(model, i, j):
        return (model.R[i, j])

    model.RM = Param(model.LINES, initialize=R_init_rule)  # Line resistance

    def X_init_rule(model, i, j):
        return (model.X[i, j])

    model.XM = Param(model.LINES, initialize=X_init_rule)  # Line resistance




    # Define Variables
    model.P = Var(model.TIMES,model.LINES, initialize=0)  # Acive power flowing in lines
    model.Q = Var(model.TIMES,model.LINES, initialize=0)  # Reacive power flowing in lines
    model.I = Var(model.TIMES,model.LINES, initialize=0)  # Current of lines

    model.SOC=Var(model.TIMES,model.NODES,initialize=model.battery_initial_soc,bounds=(model.battery_soc_min,model.battery_soc_max))
    # we set energy>0 is discharge, also only when no slack bus we put battery
    def energy_change_rule(model, time, i):
        if i not in BATTERY_NODES:
            tem = 0.0
            model.energy_change[time, i].fixed = True
        else:
            tem = 0.0
            model.energy_change[time, i].fixed = False
        return tem

    model.energy_change=Var(model.TIMES,model.NODES,initialize=energy_change_rule,bounds=(-model.battery_max_change,model.battery_max_change))

    def PS_init_rule(model, time,i):
        # for time in model.TIMES:
        if model.Tb[i].value == 0:
            temp = 0.0
            model.PS[time,i].fixed = True
        else:
            temp = 0.0
        return temp
    model.PS = Var(model.TIMES,model.NODES, initialize=PS_init_rule)  # Active power of the SS

    def QS_init_rule(model,time,i):
        # for time in model.TIMES:
        if model.Tb[i].value == 0:
            temp = 0.0
            model.QS[time,i].fixed = True
        else:
            temp = 0.0
        return temp
    model.QS = Var(model.TIMES,model.NODES, initialize=QS_init_rule)  # Reactive power of the SS
    # price init rule
    def PRICE_init_rule(model, time):
        return PRICE[time]

    model.PRICE = Param(model.TIMES, initialize=PRICE_init_rule, mutable=False)
    # Voltage of nodes
    def Voltage_init(model,time, i):
        # for time in model.TIMES:
        if model.Tb[i].value == 1:
            temp = model.Vnom
            model.V[time,i].fixed = True
        else:
            temp = model.Vnom
            model.V[time,i].fixed = False
        return temp

    model.V = Var(model.TIMES,model.NODES, initialize=Voltage_init)

    # Define Objective Function,minimize the optimal power loss. Actually, when we only have one source from the grid,
    '''Since each element of model.LINES is a tuple of two integers, 
    you will need to use two indices to access the variable indexed by model.LINES. 
    For example, if you define a variable P indexed by both model.LINES and model.TIMES, 
    you would access the value of P for the line (1,2) at time t=1 using model.P[1, (1,2)].'''
    # def act_loss(model):
    #     return (sum(sum(model.RM[i, j] * (model.I[time,(i, j)] ** 2) for i, j in model.LINES)for time in model.TIMES))

    # here we create another objective: minimizing the imported power from external grid
    # def min_power_ext_grid(model):
    #
    #     return (sum(sum(model.PS[time,node]for node in model.NODES)for time in model.TIMES))

    # Update the objective function to minimize the cost of buying energy from the external grid
    def min_cost_ext_grid(model):
        return sum(sum(model.PS[time, node] * model.PRICE[time] for node in model.NODES) for time in model.TIMES)

    model.obj = Objective(rule=min_cost_ext_grid,sense=minimize)

    # Update the objective function to earn money from battery dispatch
    # def max_benefits_dispatch_battery(model):
    #     return sum(sum(model.energy_change[time, node] * model.PRICE[time] for node in model.NODES) for time in model.TIMES)
    #
    # model.obj = Objective(rule=max_benefits_dispatch_battery,sense=maximize)



    # model.obj = Objective(rule=min_power_ext_grid)
    # model.obj = Objective(rule=act_loss)
    #we need to revise this part for adding time constraint into here.
    # %% Define Constraints
    # define soc update constraint


    def soc_update_rule(model, time, node):
        if node not in BATTERY_NODES:
            return Constraint.Skip
        if time == model.TIMES.first():
            return (model.SOC[time, node] == model.battery_initial_soc - (
                        model.energy_change[time, node] * 15.0 / 60.0) / model.battery_capacity)
        else:
            return (model.SOC[time, node] == model.SOC[model.TIMES.prev(time), node] - (
                        model.energy_change[time, node] * 15.0 / 60.0) / model.battery_capacity)

    model.constaint_soc_update=Constraint(model.TIMES,model.NODES,rule=soc_update_rule)

    # for line k consumption == injection
    def active_power_flow_rule(model, time,k):

        return (sum(model.P[time,(j, i)] for j, i in model.LINES if i == k) - sum(
            model.P[time,(i, j)] + model.RM[i, j] * (model.I[time,(i, j)] ** 2) for i, j in model.LINES if k == i) + model.PS[time,k]+model.energy_change[time,k] ==
                model.PD[time,k])

    model.active_power_flow = Constraint(model.TIMES,model.NODES, rule=active_power_flow_rule)

    def reactive_power_flow_rule(model,time, k):
        return (sum(model.Q[time,(j, i)] for j, i in model.LINES if i == k) - sum(
            model.Q[time,(i, j)] + model.XM[i, j] * (model.I[time,(i, j)] ** 2) for i, j in model.LINES if k == i) + model.QS[time,k] ==
                model.QD[time,k])

    model.reactive_power_flow = Constraint(model.TIMES,model.NODES, rule=reactive_power_flow_rule)

    ## role of voltage drop
    def voltage_drop_rule(model, time,i,j):
        return ((model.V[time, i] ** 2 - 2 * (
                    model.RM[i, j] * model.P[time, (i, j)] + model.XM[i, j] * model.Q[time, (i, j)]) - (
                        model.RM[i, j] ** 2 + model.XM[i, j] ** 2) * model.I[time, (i, j)] ** 2 - model.V[
                    time, j] ** 2 ) == 0)

    model.voltage_drop = Constraint(model.TIMES,model.LINES, rule=voltage_drop_rule)

    def define_current_rule(model, time,i, j):
        return ((model.I[time,(i, j)] ** 2) * (model.V[time,j] ** 2) == model.P[time,(i, j)] ** 2 + model.Q[time,(i, j)] ** 2)

    model.define_current = Constraint(model.TIMES,model.LINES, rule=define_current_rule)

    # here the current limit is over 0, representing that current can only from i to j, instead of versa.
    # we change this step and try to calculate it according to the result three phase one

    def current_limit_rule(model,time, i, j):
        return (0, model.I[time,(i, j)], None)
    # if we cancel this, then things to error
    model.current_limit = Constraint(model.TIMES,model.LINES, rule=current_limit_rule)

    def voltage_limit_rule(model, time, i):
        if i in BATTERY_NODES:
            return (model.Vmin, model.V[time, i], model.Vmax)
        return Constraint.Skip

    model.voltage_limit = Constraint(model.TIMES,model.NODES, rule=voltage_limit_rule)

    return model
def convert_dict_to_pd(data:dict):
    df = pd.DataFrame(columns=list(set([k[1] for k in data.keys()])))
    for key, value in data.items():
        df.loc[key[0], key[1]] = value
    # df=df.iloc[:,1:]
    # df=df.drop(df.columns[0],axis=1)
    return df
if __name__=='__main__':
    # initialize environment and network data prepared to pyomo
    env=PowerNetEnv()
    env.reset()
    net=env.net
    ppc = net._ppc
    branch = ppc['branch']
    bus=ppc['bus']
    # here we add 1, because when doing modification from PandaPower, the index of bus changed in here.
    f = np.real(branch[:, 0]).astype(int)   ## list of "from" buses
    t = np.real(branch[:, 1]).astype(int)   ## list of "to" buses
    r = branch[:, 2]
    x = branch[:, 3]
    LINES = {(f[i], t[i]) for i in range(len(f))}
    R = {(f[i], t[i]): np.real(r[i]) for i in range(len(f))}
    X = {(f[i], t[i]): np.real(x[i]) for i in range(len(f))}
    # change here and test
    # NODES = [i for i in range(bus.shape[0])]
    NODES = net.res_load.index.to_list()
    TIMES = [i for i in range(96)]
    Tb = dict()
    for i, node in enumerate(NODES):
        if i == 0:
            Tb[i] = 1
        else:
            Tb[i] = 0
    # only the last node is connected to battery
    BATTERY_NODES = {11, 15, 26,29, 33}
    # SORTED_BATTERY_NODES = [26, 15, 29, 11, 33]
    SORTED_BATTERY_NODES = [11, 15, 26,29, 33]


    # prepare netload data for all nodes
    year=2020
    month=11
    day=6
    data_manager=env.data_manager
    day_data=data_manager.select_day(year,month,day)
    # day_data shape is (96,36), we now create our active power (PD)
    active_power=day_data[:,0:34]
    qg=day_data[:,-2]
    price=day_data[:,-1]
    pv_generation=qg.reshape(96,-1)*env.pv_parameters.reshape(1,-1)
    netload=active_power-pv_generation
    PD_array=netload/1000
    PRICE = price/1000
    Data_Network = {'TIMES': TIMES, 'NODES': NODES, 'LINES': LINES, 'Tb': Tb, 'PD': PD_array, 'QD': None, 'R': R,
                    'X': X, 'BATTERY_NODES': BATTERY_NODES, 'PRICE': PRICE}
    ## prepare other parameters
    Vnom = 1.01
    Vmax = 1.05
    Vmin = 0.95
    # construct and solve model
    model=construct_opf_model(Vnom, Vmin, Vmax, Data_Network)
    # solver_path = '/home/hshengren/miniconda3/envs/caql/bin/ipopt'

    # solver = SolverFactory('ipopt',executable=solver_path)
    solver = SolverFactory('ipopt')

    solver.options['constr_viol_tol'] = 1e-6
    solver.options['acceptable_tol'] = 1e-6
    solver.options['dual_inf_tol'] = 1e-6
    solver.solve(model, tee=True,)

    objective_value = model.obj()
    print(objective_value)
    ## prepare the results
    voltage_after_control=convert_dict_to_pd(model.V.extract_values())
    active_power=convert_dict_to_pd(model.PD.extract_values())
    ext_grid_active_power=convert_dict_to_pd(model.PS.extract_values())
    ext_grid_reactive_power=convert_dict_to_pd(model.QS.extract_values())
    soc = convert_dict_to_pd(model.SOC.extract_values())
    energy_change=convert_dict_to_pd(model.energy_change.extract_values())
    print('result done')