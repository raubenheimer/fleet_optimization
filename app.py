import streamlit as st
import pandas as pd
import time
import numpy as np
from utils.gen_funcs import create_pop, evaluate_multiple, varOr, select, fuel_evaluate_multiple, fuel_create_pop, fuel_iter_gen
from utils.data_funcs import load_data, calc_cost_mats, construct_buy_sell_df, construct_fuel_df, convert_df_to_csv

# python -m streamlit run app.py
# Load dataframes
dataframes = load_data()
fuel_opt_demand, buy_cost_mat, insure_cost_mat, maintaine_cost_mat, sell_cost_mat, emissions_constraint_arr, min_veh_dict, max_evs_dict, fuel_cost_mat, fuel_cost_residual_mat, emissions_cost_mat, emissions_residual_mat, excess_range_arr = calc_cost_mats(dataframes)

def optimize_individual(n_gen):
    num_allowable_veh_sold_per_year = np.array([39., 40., 41., 42., 43., 44., 46., 46., 47., 49., 50., 51., 52., 54., 55., 57.])
    type_sum_mat = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]], dtype=np.float64)
    selling_constraint_arr = np.array([39., 40., 41., 42., 43., 44., 46., 46., 47., 49., 50., 51., 52., 54., 55., 57.])
    max_evs_allowed_arr = np.array([[9,  10, 30,  1],[9,  10, 31,  1],[9,  11, 31,  1],[38, 25, 67,  8],[40, 25, 69,  9],[41, 26, 71,  9],[81, 37, 91,  11],[82, 37, 91,  11],[84, 37, 92,  11],[93, 41, 100, 13],[95, 42, 102, 13],[97, 43, 105, 13],[98, 44, 108, 14],[102,46, 109, 14],[104,47, 112, 14],[108,49, 116, 14]])
    ev_sum_mat = np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]], dtype=np.float64)
    #######PARAMETERS#######
    # Probability of mating
    prob_mating = 0.5
    # Probability of mutating
    prob_mutating = 0.5
    # Number of offspring to produce
    lambda_ = 14000
    #Number of offspring to keep
    mu = 6000
    ########################
    # Stats for convergence check
    pop = create_pop(pop_size=mu+lambda_)
    # Evaluate the individuals
    pop_fitnesses = evaluate_multiple(pop,buy_cost_mat, insure_cost_mat, maintaine_cost_mat, fuel_cost_mat,sell_cost_mat, emissions_cost_mat, emissions_constraint_arr,type_sum_mat,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat,selling_constraint_arr,max_evs_allowed_arr,ev_sum_mat)
    # Begin the generational process
    for gen in range(1, n_gen + 1):
        # Vary the population
        offspring = varOr(pop, lambda_, prob_mating, prob_mutating, num_allowable_veh_sold_per_year, type_sum_mat)
        # Evaluate the individuals with an invalid fitness
        offspring_fitnesses = evaluate_multiple(offspring, buy_cost_mat, insure_cost_mat, maintaine_cost_mat, fuel_cost_mat,sell_cost_mat, emissions_cost_mat, emissions_constraint_arr,type_sum_mat,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat,selling_constraint_arr,max_evs_allowed_arr,ev_sum_mat)
        pop = pop + offspring
        # Select the next generation population
        pop_fitnesses = np.concatenate((pop_fitnesses, offspring_fitnesses))
        selected_idxs = select(pop_fitnesses,mu)
        pop = [pop[i] for i in selected_idxs]
        pop_fitnesses = [pop_fitnesses[i] for i in selected_idxs]
        # Update the statistics with the new population
        hof_idv = pop[0]
        min_fitness = pop_fitnesses[0]
        #print(f"Generation {gen} best fitness = {min_fitness}")
        yield gen, min_fitness, hof_idv

def optimize_fuel(indvidual,fuel_opt_demand,fuel_costs_mat,fuel_emissions_mat,emissions_constraint_arr, n_gen):
    type_sum_mat = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]], dtype=np.float64)
    indvidual = indvidual @ type_sum_mat
    max_ranges_per_size = [102000,106000,73000,118000]
    #######PARAMETERS#######
    #Number Generations
    #n_gen = 1000
    # Probability of mating
    prob_mating = 0.05
    # Probability of mutating
    prob_mutating = 0.95
    # Number of offspring to produce
    lambda_ = 4000
    # Number of offspring to keep
    mu = 1000
    # Amount of stalled generations to allow before exit
    early_stop = 100
    ########################

    fuel_opt_dict = {}

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
        # Begin the generational process
        fitness_buffer = 0
        early_stop_counter = early_stop
        for gen in range(1, n_gen + 1):            
            pop,pop_fitnesses = fuel_iter_gen(pop,pop_fitnesses,mu, lambda_, prob_mating, prob_mutating, size_cut_offs,year_fuel_costs_mat,year_fuel_emissions_mat,emissions_cap)
            hof_idv = pop[0]
            min_fitness = pop_fitnesses[0]
            #print(f"Generation {gen} best fitness = {min_fitness}")
            if min_fitness == fitness_buffer:
                early_stop_counter -= 1
                if early_stop_counter == 0:
                    break
            else:
                early_stop_counter = early_stop
            fitness_buffer = min_fitness
        best_individual = hof_idv
        fuel_opt_dict[year_idx] = best_individual
        yield year_idx + 1, min_fitness, fuel_opt_dict



st.title("Fleet Optimization Demo")
st.header("Input Data Viewer")
# Selectbox for choosing the dataframe
selected_df = st.selectbox("Select a DataFrame to view", list(dataframes.keys()))
st.dataframe(dataframes[selected_df])

st.header("Run Fleet Optimization")
st.subheader("First Optimization Step")
st.session_state.ngen = st.number_input("Select number of generations to run", min_value=1, max_value=5000, value=200)
st.session_state.fuel_ngen = st.number_input("Select number of generations to run fuel optimization", min_value=1, max_value=20000, value=1000)

st.write(f"Number: {st.session_state.ngen}")

if st.button("Start Optimization"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for current_gen, fitness, hof_idv in optimize_individual(st.session_state.ngen):
        progress_percent = current_gen/st.session_state.ngen
        progress_bar.progress(progress_percent)
        status_text.text(f"Generation {current_gen} completed with a minimum cost of {fitness}")
    st.session_state.min_fitness = fitness
    st.session_state.indvidual = hof_idv
    status_text.text(f"Optimization complete with minimum cost of {st.session_state.min_fitness}")
    sub_df = construct_buy_sell_df(st.session_state.indvidual)

    for current_year, fitness, fuel_dict in optimize_fuel(st.session_state.indvidual,fuel_opt_demand,fuel_cost_residual_mat,emissions_residual_mat,emissions_constraint_arr, st.session_state.fuel_ngen):
        progress_percent = current_year/16
        progress_bar.progress(progress_percent)
        status_text.text(f"Year {current_year+2022} fuel optimizationcompleted with a minimum cost of {fitness}")
    st.session_state.fuel_dict = fuel_dict
    status_text.text(f"Optimization complete with minimum cost of {st.session_state.min_fitness}")
    fuel_df = construct_fuel_df(st.session_state.fuel_dict)
    sub_df = pd.concat([sub_df, fuel_df]).reset_index(drop=True)
    csv = convert_df_to_csv(sub_df)
    opt_completed = True

if opt_completed:
    st.download_button(
    label="Download submission as CSV",
    data=csv,
    file_name='submission.csv',
    mime='text/csv',
    )