import numpy as np

battery_parameters={
'capacity':1000,# kW.h
'max_charge':300, # kW
'max_discharge':300, #kW
'efficiency':1,
'degradation':0, #euro/kw
'max_soc':0.8,
'min_soc':0.2,
'initial_soc':0.4}

class Battery():
    '''simulate a simple battery here'''

    def __init__(self, parameters):
        self.capacity = parameters['capacity']  # 容量
        self.max_soc = parameters['max_soc']  # max soc 0.8
        self.initial_soc = parameters['initial_soc']  # initial soc 0.4
        self.min_soc = parameters['min_soc']  # 0.2
        self.degradation = parameters['degradation']  # degradation cost 0，
        self.max_charge = parameters['max_charge']  # max charge ability
        self.max_discharge = parameters['max_discharge']  # max discharge ability
        self.efficiency = parameters['efficiency']  # charge and discharge efficiency

    def step(self, action_battery):
        energy = action_battery * self.max_charge
        updated_soc = max(self.min_soc,min(self.max_soc, (self.current_soc * self.capacity + energy * 5 / 60) / self.capacity))

        self.energy_change = (updated_soc - self.current_soc) * self.capacity * 12  # if charge, positive, if discharge, negative
        self.current_soc = updated_soc  # update capacity to current codition

    def _get_cost(self, energy):  # calculate the cost depends on the energy change
        cost = np.abs(energy)
        return cost

    def SOC(self):
        return self.current_soc

    def reset(self):
        self.current_soc = 0.4
