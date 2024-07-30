import pickle
import pandas as pd
import time
import math
import numpy as np
from deap import base, creator, tools, algorithms
from functools import partial
from datetime import datetime
import multiprocessing
from numba import njit, prange

@njit
def fuel_evaluate(individual,year_fuel_costs_mat,year_fuel_emissions_mat,emissions_cap):
    fuel_cost = 0
    emissions = 0
    for i,slot in enumerate(individual):
        slot_size = slot[0]
        slot_bucket = slot[1]
        slot_model_year = slot[2]
        slot_range = slot[3]
        slot_fuel = slot[4]
        fuel_cost += year_fuel_costs_mat[slot_model_year,(slot_size*5)+slot_fuel]*slot_range
        emissions += year_fuel_emissions_mat[slot_model_year,(slot_size*5)+slot_fuel]*slot_range
        if slot_fuel  == 0 and slot_model_year < 9:
            if slot_model_year <= 2 and slot_bucket > 0:
                fuel_cost += 500000
            elif slot_model_year <= 5 and slot_bucket > 1:
                fuel_cost += 500000
            elif slot_model_year <= 8 and slot_bucket > 2:
                fuel_cost += 500000
    if emissions > emissions_cap:
        fitness = fuel_cost + 500000
    else:
        fitness = fuel_cost
    return fitness

#### EV | DIESEL | LNG

#### Electricity |  HVO | B20 | LNG | BioLNG

@njit
def fuel_create_individual(slots, all_available_veh_arr):
    ### Size | Bucket | Year Bought | Demand | Fuel
    slots = slots.copy()
    all_available_veh_arr = all_available_veh_arr.copy()
    for slot in slots:
        slot_size = slot[0]
        veh_selection_slice = all_available_veh_arr[:,slot_size*3:slot_size*3+3]
        non_zero_indices = np.argwhere(veh_selection_slice > 0)
        if non_zero_indices.size > 0:
            idx = np.random.randint(non_zero_indices.shape[0])
            row, col = non_zero_indices[idx]
            veh_selection_slice[row, col] -= 1
        else:
            raise ValueError("Deducted more vehs than possible")
        slot[2] = row
        if col == 0:
            slot[4] = 0
        elif col == 1:
            slot[4] = np.random.randint(1,3)
        else:
            slot[4] = np.random.randint(3,5)
    
    individual = slots
    return individual

@njit                
def fuel_mutate(individual, size_cut_offs):
    fuel_map = [0,2,1,4,3]
    flip_selected = np.random.randint(0,2)
    for size_idx in range(4):
        size_slots = individual[size_cut_offs[size_idx]:size_cut_offs[size_idx+1]]
        #if flip_selected:
        # Flip Fuel Choice roll
        slot_selected = np.random.randint(0,len(size_slots))
        current_fuel = size_slots[slot_selected,4]
        size_slots[slot_selected,4] = fuel_map[current_fuel]
        #else:
        # Switch veh between slots roll
        switch_selected = np.random.choice(np.arange(1, len(size_slots)), size=2, replace=False)
        slot_1 = size_slots[switch_selected[0]].copy()
        slot_2 = size_slots[switch_selected[1]].copy()
        size_slots[switch_selected[0],2] = slot_2[2]
        size_slots[switch_selected[0],4] = slot_2[4]
        size_slots[switch_selected[1],2] = slot_1[2]
        size_slots[switch_selected[1],4] = slot_1[4]
        #print(f"size {size_idx}, flippded: {slot_selected}, switched: {switch_selected[1]},{switch_selected[0]}")
    return individual

@njit                
def fuel_mate(individual_1,individual_2, size_cut_offs):
    # size switch roll
    num_sizes_to_switch = np.random.randint(1,4)
    size_selected = np.random.randint(0,4)
    sizes_selected = np.random.choice(np.array([0,1,2,3]), size=num_sizes_to_switch, replace=False)
    for size_selected in sizes_selected:
        start_idx = size_cut_offs[size_selected]
        end_idx = size_cut_offs[size_selected+1]
        switched_segement_2 = individual_2[start_idx:end_idx].copy()
        individual_1[start_idx:end_idx] = switched_segement_2
    return individual_1

@njit
def fuel_create_pop(pop_size,slots, all_available_veh_arr):
    pop = []
    for _ in range(pop_size):
        pop.append(fuel_create_individual(slots, all_available_veh_arr))
    return pop

@njit(parallel=True)
def fuel_evaluate_multiple(pop,year_fuel_costs_mat,year_fuel_emissions_mat,emissions_cap):
    fitnesses = np.zeros(len(pop))
    for i in prange(len(pop)):
        fitnesses[i] = fuel_evaluate(pop[i],year_fuel_costs_mat,year_fuel_emissions_mat,emissions_cap)
    return fitnesses

@njit
def fuel_varOr(population, lambda_, cxpb, mutpb, size_cut_offs):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")
    pop_size = len(population)
    offspring = []
    for i in range(lambda_):
        op_choice = np.random.random()
        # Apply crossover
        if op_choice < cxpb:
            mate_idxs = np.random.choice(pop_size, 2, replace=False)
            #ind1, ind2 = [np.copy(i) for i in np.random.choice(population, 2, replace=False)]
            ind1 = population[mate_idxs[0]]
            ind2 = population[mate_idxs[1]]
            ind1 = fuel_mate(ind1, ind2, size_cut_offs)
            offspring.append(ind1)
        # Apply mutation    
        elif op_choice < cxpb + mutpb:        
            mutate_idx = np.random.randint(0, pop_size)
            ind = np.copy(population[mutate_idx])
            ind = fuel_mutate(ind, size_cut_offs)
            offspring.append(ind)
        # Apply reproduction    
        else:                
            rep_idx = np.random.choice(pop_size)
            ind = np.copy(population[rep_idx])
            offspring.append(ind)
    return offspring

@njit
def fuel_select(pop_fitnesses,mu):
    sorted_indices = np.argsort(pop_fitnesses)    
    return sorted_indices[:mu]

@njit
def fuel_iter_gen(pop, pop_fitnesses, mu, lambda_, prob_mating, prob_mutating, size_cut_offs,year_fuel_costs_mat,year_fuel_emissions_mat,emissions_cap):
    # Vary the population
    offspring = fuel_varOr(pop, lambda_, prob_mating, prob_mutating, size_cut_offs)
    # Evaluate the individuals with an invalid fitness
    offspring_fitnesses = fuel_evaluate_multiple(offspring,year_fuel_costs_mat,year_fuel_emissions_mat,emissions_cap)
    pop = pop + offspring
    # Select the next generation population
    pop_fitnesses = np.concatenate((pop_fitnesses, offspring_fitnesses))
    selected_idxs = fuel_select(pop_fitnesses,mu)
    pop = [pop[i] for i in selected_idxs]
    pop_fitnesses = pop_fitnesses[selected_idxs] #selected_idxs
    return pop,pop_fitnesses

def main(indvidual,fuel_opt_demand,fuel_costs_mat,fuel_emissions_mat,emissions_constraint_arr):
    type_sum_mat = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]], dtype=np.float64)
    indvidual = indvidual @ type_sum_mat
    max_ranges_per_size = [102000,106000,73000,118000]
    veh_type_map = ["BEV","Diesel","Diesel","LNG","LNG"]
    fuel_map = ["Electricity","HVO","B20","LNG","BioLNG"]
    #######PARAMETERS#######
    #Number Generations
    n_gen = 1000
    # Probability of mating
    prob_mating = 0.05
    # Probability of mutating
    prob_mutating = 0.95
    # Number of offspring to produce
    lambda_ = 4000
    #Number of offspring to keep
    mu = 1000

    fuel_opt_dict = {}
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    for year_idx in range(16):
        # Individual setup
        depth_slice = indvidual[year_idx]
        year_slots_list = []
        size_cut_offs = np.zeros(5)
        year_emissions_cap = emissions_constraint_arr[year_idx]
        for size_idx in range(4):
            size = size_idx + 1
            bucket_demands = fuel_opt_demand[year_idx,size_idx*4:size_idx*4+4]
            size_range = max_ranges_per_size[size_idx]
            slots_per_bucket_arr = np.ceil(bucket_demands/size_range).astype(int)
            residuals = np.mod(bucket_demands, size_range)
            for bucket_idx,num_bucket_slots in enumerate(slots_per_bucket_arr):
                bucket = bucket_idx + 1
                for i in range(num_bucket_slots-1):
                    year_slots_list.append([size_idx,bucket_idx,0,size_range,0])
                year_slots_list.append([size_idx,bucket_idx,0,residuals[bucket_idx],0])
        fuel_slots = np.array(year_slots_list)
        # Get cut off idxs
        size_cut_offs = np.zeros(5, dtype=int)
        current_size = 0
        idx_counter = 0
        for slot in fuel_slots:
            if slot[0] != current_size:
                size_cut_offs[current_size+1] = idx_counter #+ 1
                current_size += 1
            idx_counter += 1
        size_cut_offs[4] = len(fuel_slots)

        year_fuel_costs_mat=fuel_costs_mat[year_idx,:,:]
        year_fuel_emissions_mat=fuel_emissions_mat[year_idx,:,:]
        emissions_cap=year_emissions_cap
        
        print(f"Year: {year_idx+2023}")
        pop = fuel_create_pop(pop_size=mu+lambda_,slots=fuel_slots, all_available_veh_arr=depth_slice[:year_idx+1,:])
        # Evaluate the individuals
        pop_fitnesses = fuel_evaluate_multiple(pop,year_fuel_costs_mat,year_fuel_emissions_mat,emissions_cap)
        early_stop_counter = 100
        # Begin the generational process
        fitness_buffer = 0
        for gen in range(1, n_gen + 1):            
            pop,pop_fitnesses = fuel_iter_gen(pop,pop_fitnesses,mu, lambda_, prob_mating, prob_mutating, size_cut_offs,year_fuel_costs_mat,year_fuel_emissions_mat,emissions_cap)
            # Update the statistics with the new population
            hof_idv = pop[0]
            min_fitness = pop_fitnesses[0]
            print(f"Generation {gen} best fitness = {min_fitness}")
            if min_fitness == fitness_buffer:
                early_stop_counter -= 1
                if early_stop_counter == 0:
                    break
            else:
                early_stop_counter = 100
            fitness_buffer = min_fitness
        best_individual = hof_idv
        fuel_opt_dict[year_idx] = best_individual

    # date_string = datetime.now().strftime('%Y-%m-%d %H')
    # with open(f'HOF/fuel_{date_string}.pkl', 'wb') as file:
    #     pickle.dump(fuel_opt_dict, file)

    headers = ["Year","ID","Num_Vehicles","Type","Fuel","Distance_bucket","Distance_per_vehicle(km)"]
    rows_list = []
    for year_idx in fuel_opt_dict:
        for slot in fuel_opt_dict[year_idx]:
            year = year_idx + 2023
            veh_id = f"{veh_type_map[slot[4]]}_S{slot[0]+1}_{slot[2]+2023}"
            num_veh = 1
            sub_type = "Use"
            fuel = fuel_map[slot[4]]
            distance_bucket = f"D{slot[1]+1}"
            distance_veh = slot[3]
            rows_list.append([year,veh_id,num_veh,sub_type,fuel,distance_bucket,distance_veh])
    fuel_opt_df = pd.DataFrame(rows_list,columns=headers)
    fuel_opt_df.to_csv("fuel_opt.csv",index=False)
        
        


if __name__ == '__main__':
    indv_name = "2024-06-28 12_11_14.npy"
    indvidual = np.load(f"{indv_name}")

    # Load in orginal datasets REPLACE WITH COPIES THAT ARE ALREADY IN MEMORY
    vehicles_fuels_df = pd.read_csv("dataset/vehicles_fuels.csv")
    fuels_df = pd.read_csv("dataset/fuels.csv")
    demand_df = pd.read_csv("dataset/demand.csv")
    vehicles_df = pd.read_csv("dataset/vehicles.csv")
    cost_profiles_df = pd.read_csv("dataset/cost_profiles.csv")
    carbon_emissions_df = pd.read_csv("dataset/carbon_emissions.csv")

    range_arr = np.array([102000,102000,102000,102000,102000,106000,106000,106000,106000,106000,118000,118000,118000,118000,118000,73000,73000,730007,73000,73000,])

    # fuel_opt_demand
    fuel_demand_df = demand_df.copy()
    fuel_demand_df["ID"] = fuel_demand_df["Size"] + "_" + fuel_demand_df["Distance"]
    fuel_demand_df = fuel_demand_df.drop(["Size","Distance"],axis=1)
    fuel_demand_df = fuel_demand_df.pivot(index='Year', columns='ID', values='Demand (km)').reset_index()
    fuel_demand_df = fuel_demand_df.drop(["Year"],axis=1)
    fuel_opt_demand =np.array(fuel_demand_df.values)

    # emissions_constraint_arr
    emissions_constraint_arr = np.array(carbon_emissions_df["Carbon emission CO2/kg"])

    # fuel_cost_mat & fuel_cost_residual_mat
    #### Electricity |  HVO | B20 | LNG | BioLNG
    fuel_consumption_ref_df = vehicles_fuels_df.copy()
    fuel_consumption_ref_df["Model Year"] = vehicles_fuels_df["ID"].apply(lambda x: x.split("_")[2])
    fuel_consumption_ref_df["Size"] = vehicles_fuels_df["ID"].apply(lambda x: x.split("_")[1])
    fuel_consumption_ref_df["Fuel Size"] = fuel_consumption_ref_df["Fuel"] + "_" + fuel_consumption_ref_df["Size"]
    fuel_consumption_ref_df = fuel_consumption_ref_df.pivot(index='Model Year', columns='Fuel Size', values='Consumption (unit_fuel/km)').reset_index()
    fuel_consumption_ref_df = fuel_consumption_ref_df[['Electricity_S1','HVO_S1','B20_S1','LNG_S1','BioLNG_S1','Electricity_S2','HVO_S2','B20_S2','LNG_S2','BioLNG_S2','Electricity_S3','HVO_S3','B20_S3','LNG_S3','BioLNG_S3','Electricity_S4', 'HVO_S4', 'B20_S4', 'LNG_S4','BioLNG_S4']]
    fuel_consumption_ref_arr = np.array(fuel_consumption_ref_df.values)
    fuel_cost_mat = np.zeros((16,16,20))
    for year_idx in range(16):
        year = year_idx +2023
        yearly_fuel_costs_df = fuels_df[fuels_df["Year"] == year]
        yearly_fuel_costs_df = yearly_fuel_costs_df.pivot(index='Year', columns='Fuel', values='Cost ($/unit_fuel)').reset_index()
        yearly_fuel_costs_df = yearly_fuel_costs_df[["Electricity","HVO","B20","LNG","BioLNG"]]
        yearly_fuel_costs_arr = np.array(yearly_fuel_costs_df.values)
        repeated_array = np.tile(yearly_fuel_costs_arr, 4)
        fuel_cost_mat[year_idx,:year_idx+1,:] = fuel_consumption_ref_arr[:year_idx+1] * repeated_array
    fuel_cost_residual_mat = fuel_cost_mat.copy()
    for depth in range(len(fuel_cost_mat)):
        for row in range(len(fuel_cost_mat)):
            fuel_cost_mat[depth,row,:] = fuel_cost_mat[depth,row,:]*range_arr
            if(row==depth):
                break

    # emissions_cost_mat & emissions_residual_mat
    #### Electricity |  HVO | B20 | LNG | BioLNG
    fuel_consumption_ref_df = vehicles_fuels_df.copy()
    fuel_consumption_ref_df["Model Year"] = vehicles_fuels_df["ID"].apply(lambda x: x.split("_")[2])
    fuel_consumption_ref_df["Size"] = vehicles_fuels_df["ID"].apply(lambda x: x.split("_")[1])
    fuel_consumption_ref_df["Fuel Size"] = fuel_consumption_ref_df["Fuel"] + "_" + fuel_consumption_ref_df["Size"]
    fuel_consumption_ref_df = fuel_consumption_ref_df.pivot(index='Model Year', columns='Fuel Size', values='Consumption (unit_fuel/km)').reset_index()
    fuel_consumption_ref_df = fuel_consumption_ref_df[['Electricity_S1','HVO_S1','B20_S1','LNG_S1','BioLNG_S1','Electricity_S2','HVO_S2','B20_S2','LNG_S2','BioLNG_S2','Electricity_S3','HVO_S3','B20_S3','LNG_S3','BioLNG_S3','Electricity_S4', 'HVO_S4', 'B20_S4', 'LNG_S4','BioLNG_S4']]
    fuel_consumption_ref_arr = np.array(fuel_consumption_ref_df.values)
    emissions_cost_mat = np.zeros((16,16,20))
    for year_idx in range(16):
        year = year_idx +2023
        yearly_emissions_df = fuels_df[fuels_df["Year"] == year]
        yearly_emissions_df = yearly_emissions_df.pivot(index='Year', columns='Fuel', values='Emissions (CO2/unit_fuel)').reset_index()
        yearly_emissions_df = yearly_emissions_df[["Electricity","HVO","B20","LNG","BioLNG"]]
        yearly_emissions_arr = np.array(yearly_emissions_df.values)
        repeated_array = np.tile(yearly_emissions_arr, 4)
        emissions_cost_mat[year_idx,:year_idx+1,:] = fuel_consumption_ref_arr[:year_idx+1] * repeated_array
    emissions_residual_mat = emissions_cost_mat.copy() 
    for depth in range(len(emissions_cost_mat)):
        for row in range(len(emissions_cost_mat)):
            emissions_cost_mat[depth,row,:] = emissions_cost_mat[depth,row,:]*range_arr
            if(row==depth):
                break
        

    main(indvidual,fuel_opt_demand,fuel_cost_residual_mat,emissions_residual_mat,emissions_constraint_arr)