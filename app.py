import streamlit as st
import pandas as pd
import time
import numpy as np
import math
from deap import base, creator, tools, algorithms
from functools import partial
from utils.gen_funcs import get_sold_arr, create_wrapper, create_individual, calc_fuel_cost, evaluate, mate, mutate
from utils.data_funcs import load_data, calc_cost_mats


# Load dataframes
dataframes = load_data()
fuel_opt_demand, buy_cost_mat, insure_cost_mat, maintaine_cost_mat, sell_cost_mat, emissions_constraint_arr, min_veh_dict, max_evs_dict, fuel_cost_mat, fuel_cost_residual_mat, emissions_cost_mat, emissions_residual_mat, excess_range_arr = calc_cost_mats(dataframes)

def init_toolbox():
    min_veh_req_arr = np.array([[ 73., 34., 81., 11.],[ 74., 34., 83., 11.],[ 76., 35., 84., 11.],[ 78., 35., 87., 11.],[ 81., 36., 89., 12.],[ 83., 37., 91., 12.],[ 86., 39., 95., 12.],[ 87., 39., 95., 12.],[ 90., 39., 96., 12.],[ 93., 41.,100., 13.],[ 95., 42.,102., 13.],[ 97., 43.,105., 13.],[ 98., 44.,108., 14.],[102., 46.,109., 14.],[104., 47.,112., 14.],[108., 49.,116., 14.],])
    max_evs_allowed_arr = np.array([[9,  10, 30,  1],[9,  10, 31,  1],[9,  11, 31,  1],[38, 25, 67,  8],[40, 25, 69,  9],[41, 26, 71,  9],[81, 37, 91,  11],[82, 37, 91,  11],[84, 37, 92,  11],[93, 41, 100, 13],[95, 42, 102, 13],[97, 43, 105, 13],[98, 44, 108, 14],[102,46, 109, 14],[104,47, 112, 14],[108,49, 116, 14]])
    num_allowable_veh_sold_per_year = np.array([39., 40., 41., 42., 43., 44., 46., 46., 47., 49., 50., 51., 52., 54., 55., 57.])
    type_sum_mat = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]], dtype=np.float64)
    ev_sum_mat = np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]], dtype=np.float64)
    selling_constraint_arr = np.array([39., 40., 41., 42., 43., 44., 46., 46., 47., 49., 50., 51., 52., 54., 55., 57.])
    toolbox = base.Toolbox()
    # Individual setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    # Individual generator
    toolbox.register("individual", tools.initIterate, creator.Individual, create_wrapper)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Register Fitness function
    toolbox.register("evaluate", partial(evaluate, buy_cost_mat=buy_cost_mat, insure_cost_mat=insure_cost_mat,maintaine_cost_mat=maintaine_cost_mat,fuel_cost_mat=fuel_cost_mat,sell_cost_mat=sell_cost_mat,emissions_cost_mat=emissions_cost_mat,emissions_constraint_arr=emissions_constraint_arr,type_sum_mat=type_sum_mat,excess_range_arr=excess_range_arr,fuel_cost_residual_mat=fuel_cost_residual_mat,emissions_residual_mat=emissions_residual_mat,selling_constraint_arr=selling_constraint_arr,max_evs_allowed_arr=max_evs_allowed_arr,ev_sum_mat=ev_sum_mat))
    # Register the mustate function
    toolbox.register("mutate", mutate, min_veh_req_arr=min_veh_req_arr, max_evs_allowed_arr=max_evs_allowed_arr, num_allowable_veh_sold_per_year=num_allowable_veh_sold_per_year,type_sum_mat=type_sum_mat)
    # Register the crossover function
    toolbox.register("mate", mate)
    # Register the selection function
    toolbox.register("select", tools.selBest)
    return toolbox

def optimization_task(params):
    for i in range(10):
        time.sleep(1)  # Simulate a time-consuming task
        yield f"Progress: {i * 10}%"

def optimize_individual(toolbox, n_gen):
    #######PARAMETERS#######
    #Population Size
    pop_size = 20000
    #Number Generations
    #n_gen = 100
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
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if hof is not None:
        hof.update(pop)
    record = stats.compile(pop) if stats is not None else {}
    # Begin the generational process
    for gen in range(1, n_gen + 1):
        # Vary the population
        offspring = algorithms.varOr(pop, toolbox, lambda_, prob_mating, prob_mutating)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)        
        # the slow part
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if hof is not None:
            hof.update(offspring)
        # Select the next generation population
        pop[:] = toolbox.select(pop + offspring, mu)
        # Update the statistics with the new population
        record = stats.compile(pop) if stats is not None else {}
        min_fitness = record["min"][0]
        #print(f"Generation {gen} best fitness = {min_fitness}")
        yield gen, min_fitness

st.title("Fleet Optimization Demo")
st.header("Input Data Viewer")
# Selectbox for choosing the dataframe
selected_df = st.selectbox("Select a DataFrame to view", list(dataframes.keys()))
st.dataframe(dataframes[selected_df])

st.header("Run Fleet Optimization")
st.subheader("First Optimization Step")
st.session_state.ngen = st.number_input("Select number of generations to run", min_value=0, max_value=100, value=50)
st.write(f"Number: {st.session_state.ngen}")

if st.button("Start Optimization"):
    toolbox = init_toolbox()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for current_gen, fitness in optimize_individual(toolbox, st.session_state.ngen):
        progress_percent = current_gen/st.session_state.ngen
        progress_bar.progress(progress_percent)
        status_text.text(f"Generation {current_gen} completed with fitness {fitness}")
    
    status_text.text("Optimization complete!")