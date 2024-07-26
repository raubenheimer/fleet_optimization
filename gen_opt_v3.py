import pickle
import pandas as pd
import time
import math
import cvxpy as cp
import numpy as np
from deap import base, creator, tools, algorithms
from functools import partial
from datetime import datetime
from multiprocessing import Pool
from numba import jit, njit,prange


'''
At the "last year of period of evaluation" our eval script automatically sells everything remaining in your fleet and subtracts it from your cost. And yes this overrides the corner case of "buying in the last year and being able to sell only 20% in a year".
"Last year of period of evaluation" for public is 2028 and private is 2038.
'''



np.set_printoptions(threshold=np.inf,linewidth=100)


@njit
def get_sold_arr(indv):
    num_veh_sold_mat = np.zeros((16,16,12), dtype=np.float64)
    for depth in range(len(indv)-1):
        for row in range(len(indv)):
            num_veh_sold_mat[depth,row,0] = indv[depth+1, row, 0] - indv[depth, row, 0]
            num_veh_sold_mat[depth,row,1] = indv[depth+1, row, 1] - indv[depth, row, 1] + indv[depth+1, row, 2] - indv[depth, row, 2]
            num_veh_sold_mat[depth,row,2] = indv[depth+1, row, 3] - indv[depth, row, 3] + indv[depth+1, row, 4] - indv[depth, row, 4]
            num_veh_sold_mat[depth,row,3] = indv[depth+1, row, 5] - indv[depth, row, 5]
            num_veh_sold_mat[depth,row,4] = indv[depth+1, row, 6] - indv[depth, row, 6] + indv[depth+1, row, 7] - indv[depth, row, 7]
            num_veh_sold_mat[depth,row,5] = indv[depth+1, row, 8] - indv[depth, row, 8] + indv[depth+1, row, 9] - indv[depth, row, 9]
            num_veh_sold_mat[depth,row,6] = indv[depth+1, row, 10] - indv[depth, row, 10]
            num_veh_sold_mat[depth,row,7] = indv[depth+1, row, 11] - indv[depth, row, 11] + indv[depth+1, row, 12] - indv[depth, row, 12]
            num_veh_sold_mat[depth,row,8] = indv[depth+1, row, 13] - indv[depth, row, 13] + indv[depth+1, row, 14] - indv[depth, row, 14]
            num_veh_sold_mat[depth,row,9] = indv[depth+1, row, 15] - indv[depth, row, 15]
            num_veh_sold_mat[depth,row,10] = indv[depth+1, row, 16] - indv[depth, row, 16] + indv[depth+1, row, 17] - indv[depth, row, 17]
            num_veh_sold_mat[depth,row,11] = indv[depth+1, row, 18] - indv[depth, row, 18] + indv[depth+1, row, 19] - indv[depth, row, 19]
            if(depth == row):
                break
    return num_veh_sold_mat

#@njit
def indv_checker(individual,type_sum_mat):
    individual_type = np.zeros((16,16,12))
    number_sold_mat = get_sold_arr(individual)
    for depth_idx in range(len(individual_type[:,:,:])):
        individual_type[depth_idx] = individual[depth_idx] @ type_sum_mat
    min_veh_req_arr = np.array([[ 73., 34., 81., 11.],[ 74., 34., 83., 11.],[ 76., 35., 84., 11.],[ 78., 35., 87., 11.],[ 81., 36., 89., 12.],[ 83., 37., 91., 12.],[ 86., 39., 95., 12.],[ 87., 39., 95., 12.],[ 90., 39., 96., 12.],[ 93., 41.,100., 13.],[ 95., 42.,102., 13.],[ 97., 43.,105., 13.],[ 98., 44.,108., 14.],[102., 46.,109., 14.],[104., 47.,112., 14.],[108., 49.,116., 14.],])
    size_sum_matrix = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],
                                [0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],
                                [0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1]], dtype=np.float64)
    max_evs_allowed_arr = np.array([[9,  10, 30,  1],[9,  10, 31,  1],[9,  11, 31,  1],[38, 25, 67,  8],[40, 25, 69,  9],[41, 26, 71,  9],[81, 37, 91,  11],[82, 37, 91,  11],[84, 37, 92,  11],[93, 41, 100, 13],[95, 42, 102, 13],[97, 43, 105, 13],[98, 44, 108, 14],[102,46, 109, 14],[104,47, 112, 14],[108,49, 116, 14]])
    ev_sum_matrix = np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,0,0],
                              [0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0],
                              [0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], dtype=np.float64)
    num_allowable_veh_sold_per_year = np.array([39., 40., 41., 42., 43., 44., 46., 46., 47., 49., 50., 51., 52., 54., 55., 57.])
    sold_sum_matrix = np.array([[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]], dtype=np.float64)
    num_veh_sold_per_year = np.zeros_like(num_allowable_veh_sold_per_year)
    num_veh_per_size = np.zeros_like(min_veh_req_arr)
    num_evs_per_size = np.zeros_like(max_evs_allowed_arr)
    
    for depth in range(len(individual_type)):
        num_veh_per_size[depth,:] = np.sum(individual_type[depth,:,:] @ size_sum_matrix,axis=0)
        num_evs_per_size[depth,:] = np.sum(individual_type[depth,:,:] @ ev_sum_matrix,axis=0)
        num_veh_sold_per_year[depth] = (np.sum(number_sold_mat[depth,:,:] @ sold_sum_matrix,axis=1)[0])*-1
    
    # CHECK min veh requirment
    if not np.array_equal(min_veh_req_arr,num_veh_per_size):
        print("OOOOOOOOOOOOOOOOOOOOOO")
        raise ValueError(f"Number of total veh not satisfied")
    # CHECK max evs
    evs_comparison = num_evs_per_size <= max_evs_allowed_arr
    all_years_evs_valid = np.all(evs_comparison)
    #if not all_years_evs_valid:
    #    raise ValueError("Number ev greater than allowed")
    # CHECK fleet sold <= 20%
    sold_comparison = num_veh_sold_per_year <= num_allowable_veh_sold_per_year
    all_years_sold_valid = np.all(sold_comparison)
    if not all_years_sold_valid:
        raise ValueError("Selling constraint not respected")
    # CHECK selling veh pos
    sold_pos = np.any(number_sold_mat.ravel() > 0) #doing it like this for Numba
    if sold_pos:
        #flat_index = np.argmax(number_sold_mat > 0)
        #index_3d = np.unravel_index(flat_index, number_sold_mat.shape)
        raise ValueError(f"Sold value xx is pos")
    if len(np.nonzero(individual < 0)[0]):
        print(individual)
        raise ValueError("Neg values")


def create_wrapper():
    indv = creator.Individual(create_individual())
    return indv

@njit
def create_individual():
    num_years = 16
    size_list = ["S1","S2","S3","S4"]
    min_veh_req_arr = np.array([[ 73., 34., 81., 11.],[ 74., 34., 83., 11.],[ 76., 35., 84., 11.],[ 78., 35., 87., 11.],[ 81., 36., 89., 12.],[ 83., 37., 91., 12.],[ 86., 39., 95., 12.],[ 87., 39., 95., 12.],[ 90., 39., 96., 12.],[ 93., 41.,100., 13.],[ 95., 42.,102., 13.],[ 97., 43.,105., 13.],[ 98., 44.,108., 14.],[102., 46.,109., 14.],[104., 47.,112., 14.],[108., 49.,116., 14.],])
    max_evs_allowed_arr = np.array([[9,  10, 30,  1],[9,  10, 31,  1],[9,  11, 31,  1],[38, 25, 67,  8],[40, 25, 69,  9],[41, 26, 71,  9],[81, 37, 91,  11],[82, 37, 91,  11],[84, 37, 92,  11],[93, 41, 100, 13],[95, 42, 102, 13],[97, 43, 105, 13],[98, 44, 108, 14],[102,46, 109, 14],[104,47, 112, 14],[108,49, 116, 14]])
    size_sum_matrix = np.array([[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]], dtype=np.float64)
    num_allowable_veh_sold_per_year = np.array([39, 40, 41, 42, 43, 44, 46, 46, 47, 49, 50, 51, 52, 54, 55, 57], dtype=np.int64)
    individual = np.zeros((num_years,num_years,20), dtype=np.float64)
    first_year = 2023
    # INIT 2023
    year = 2023
    count = 0
    first_year_arr = np.zeros(20)
    for size in range(len(size_list)):
        num_veh_of_size = int(min_veh_req_arr[year-first_year][size])
        max_evs_of_size = max_evs_allowed_arr[year-first_year][size]
        number_of_evs = np.random.randint(0, max_evs_of_size+1)
        max_num_diesels = num_veh_of_size - number_of_evs
        num_total_diesels = np.random.randint(0, max_num_diesels+1)        
        number_of_hvo = np.random.randint(0, num_total_diesels+1)
        number_of_b20 = num_total_diesels - number_of_hvo
        num_total_lngs = num_veh_of_size - num_total_diesels - number_of_evs 
        number_of_lng = np.random.randint(0, num_total_lngs+1)
        number_of_biolng = num_total_lngs - number_of_lng
        # Electricity |  HVO | B20 | LNG | BioLNG
        first_year_arr[count] = number_of_evs
        count += 1
        first_year_arr[count] = number_of_hvo
        count += 1
        first_year_arr[count] = number_of_b20
        count += 1
        first_year_arr[count] = number_of_lng
        count += 1
        first_year_arr[count] = number_of_biolng
        count += 1
    individual[0, 0, :] = first_year_arr
    
    # YEAR >= 2024
    while(year< (2022 + num_years)):
        year = year + 1
        depth_index = year - first_year
        previous_year_mat = individual[depth_index-1, :, :] 
        previous_year_veh_num_detailed = np.sum(previous_year_mat, axis=0) # number of each veh class for previous year
        # Sell Rolls
        num_max_sellable_veh = num_allowable_veh_sold_per_year[depth_index-1]
        actually_selling = np.random.randint(0, num_max_sellable_veh)
        if actually_selling > 0:
            breakpoints = np.sort(np.random.randint(0, actually_selling, 20 - 1))
            breakpoints = np.concatenate((np.array([0]), breakpoints, np.array([actually_selling])))
            sold_for_detailed = np.diff(breakpoints.astype(np.float64))
        else:
            sold_for_detailed = np.zeros(20, dtype=np.float64)
        for i in range(20):
            if sold_for_detailed[i] > previous_year_veh_num_detailed[i]:
                sold_for_detailed[i] = previous_year_veh_num_detailed[i]
        num_veh_after_sale_detailed = previous_year_veh_num_detailed-sold_for_detailed
        num_veh_after_sale_per_size = size_sum_matrix @ num_veh_after_sale_detailed
        #Detailed selling logic
        deduction_stepped_vec = -1*sold_for_detailed
        individual[depth_index, :, :] = individual[depth_index-1, :, :]
        count = 0
        while(count<depth_index):
            individual[depth_index, count, :] = individual[depth_index, count, :] + deduction_stepped_vec
            deduction_stepped_vec = np.where(individual[depth_index, count, :] > 0, 0, individual[depth_index, count, :])
            individual[depth_index, count, :][individual[depth_index, count, :] < 0] = 0
            count += 1
        count = 0
        # Buy Rolls
        for i, size in enumerate(size_list):
            required_veh_of_size = int(min_veh_req_arr[year-first_year][i] - num_veh_after_sale_per_size[i])
            max_evs_of_size = max_evs_allowed_arr[year-first_year][i]
            current_amount_evs = num_veh_after_sale_detailed[i*5]
            ev_gap_to_max = max_evs_of_size - current_amount_evs
            # Buy Rolls
            max_num_of_evs = int(min(required_veh_of_size, ev_gap_to_max))
            number_of_evs = np.random.randint(0, max_num_of_evs+1)
            number_of_diesels_total = np.random.randint(0, required_veh_of_size - number_of_evs + 1)
            number_of_hvo = np.random.randint(0,number_of_diesels_total+1)
            number_of_b20 = number_of_diesels_total - number_of_hvo
            number_of_lngs_total = required_veh_of_size - number_of_diesels_total - number_of_evs
            number_of_lng = np.random.randint(0,number_of_lngs_total+1)
            number_of_biolng = number_of_lngs_total - number_of_lng
            # Electricity |  HVO | B20 | LNG | BioLNG
            individual[depth_index, depth_index, count] = number_of_evs
            count += 1
            individual[depth_index, depth_index, count] = number_of_hvo
            count += 1
            individual[depth_index, depth_index, count] = number_of_b20
            count += 1
            individual[depth_index, depth_index, count] = number_of_lng
            count += 1
            individual[depth_index, depth_index, count] = number_of_biolng
            count += 1
    #Shuffel Fuels
    for depth_idx in range(1,16):
        for row_idx in range(0,depth_idx):
            for size_idx in range(4):
                total_diesels = int(np.sum(individual[depth_idx,row_idx,size_idx*5+1:size_idx*5+3]))
                total_lng = int(np.sum(individual[depth_idx,row_idx,size_idx*5+3:size_idx*5+5]))
                if total_diesels > 0:
                    new_hvo = np.random.randint(0,total_diesels+1)
                    new_b20 = total_diesels - new_hvo
                    individual[depth_idx,row_idx,size_idx*5+1] = new_hvo
                    individual[depth_idx,row_idx,size_idx*5+2] = new_b20
                if total_lng > 0:
                    new_lng = np.random.randint(0,total_lng+1)
                    new_biolng = total_lng - new_lng
                    individual[depth_idx,row_idx,size_idx*5+3] = new_lng
                    individual[depth_idx,row_idx,size_idx*5+4] = new_biolng

    return individual

@njit
def calc_fuel_cost(individual,fuel_cost_mat,emissions_cost_mat, emissions_constraint_arr,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat):
    fuel_cost_total = np.sum(individual*fuel_cost_mat)
    emissions_total = np.sum(np.sum(individual * emissions_cost_mat, axis=2), axis=1)
    over_pen = 0
    # CALC RESIDUALS
    fuel_cost_residual = 0
    for year_idx in range(16):
        year_emissions = 0
        for size_idx in range(4):
            residuals_arr = excess_range_arr[year_idx,size_idx*4:size_idx*4+4]
            bucket_ass_arr = np.zeros(4)
            emissions_ass_arr = np.zeros(4)
            veh_avail_mat = individual[year_idx,:year_idx+1,size_idx*5:size_idx*5+5]
            veh_avail_arr = veh_avail_mat.flatten()
            fuels_mat = fuel_cost_residual_mat[year_idx,:year_idx+1,size_idx*5:size_idx*5+5]
            fuels_arr = fuels_mat.flatten()
            emissions_mat = emissions_residual_mat[year_idx,:year_idx+1,size_idx*5:size_idx*5+5]
            emissions_arr = emissions_mat.flatten()
            veh_avail_indexes = np.nonzero(veh_avail_arr > 0)[0]
            fuels_avail = fuels_arr[veh_avail_indexes]
            veh_avail_arr = veh_avail_arr[veh_avail_indexes]
            emissions_avail = emissions_arr[veh_avail_indexes]
            fuel_sorted_idx_arr = np.argsort(fuels_avail)
            buckets_to_assign = 4
            bucket_idx = 0
            fuel_idx = 0
            ### NBNBNBNBNB ADD EV BUCKET LOGIC
            while bucket_idx < 4:
                fuel_cost = fuels_avail[fuel_sorted_idx_arr[fuel_idx]]
                num_veh = int(veh_avail_arr[fuel_sorted_idx_arr[fuel_idx]])
                emissions = emissions_avail[fuel_sorted_idx_arr[fuel_idx]] 
                if num_veh < 4 - bucket_idx:
                    bucket_ass_arr[:num_veh] = fuel_cost
                    emissions_ass_arr[:num_veh] = emissions
                    bucket_idx += num_veh
                    fuel_idx += 1
                else:
                    bucket_ass_arr[:] = fuel_cost
                    emissions_ass_arr[:] = emissions
                    break
            year_size_residual_cost = np.sum(bucket_ass_arr*residuals_arr)
            year_size_residual_emission = np.sum(emissions_ass_arr*residuals_arr)
            fuel_cost_residual += year_size_residual_cost
            year_emissions += year_size_residual_emission
        if (emissions_total[year_idx] - year_emissions) > emissions_constraint_arr[year_idx]:
           over_pen += 100000000
    return fuel_cost_total - fuel_cost_residual + over_pen


@njit
def evaluate(individual, buy_cost_mat, insure_cost_mat, maintaine_cost_mat, fuel_cost_mat,sell_cost_mat, emissions_cost_mat, emissions_constraint_arr,type_sum_mat,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat,selling_constraint_arr,max_evs_allowed_arr,ev_sum_mat):
    # CHECK SELLING LIMITS AND PENILIZE
    sold_arr = get_sold_arr(individual)
    #ev_sum_mat = np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]], dtype=np.float64)
    #selling_constraint_arr = np.array([39., 40., 41., 42., 43., 44., 46., 46., 47., 49., 50., 51., 52., 54., 55., 57.])
    selling_actual_arr = -np.sum(np.sum(sold_arr,axis=1),axis=1) 
    selling_comparison = selling_actual_arr > selling_constraint_arr
    selling_violations_count = np.sum(selling_comparison) 
    if selling_violations_count > 0:
        fitness = 2760000000.0 + selling_violations_count * 100000000.0
        return (fitness,)
    # CHECK EVS

    #FUEL AND EMISSIONS
    fuel_cost = calc_fuel_cost(individual,fuel_cost_mat,emissions_cost_mat, emissions_constraint_arr,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat)
    if fuel_cost == 3000000000.0:
        return (fuel_cost,)
    indv_types = np.zeros((16,16,12))
    total_evs_per_year = np.zeros((16,4))
    for depth_idx in range(len(indv_types[:,:,:])):
        indv_types[depth_idx] = individual[depth_idx] @ type_sum_mat
        total_evs_per_year[depth_idx] = np.sum(individual[depth_idx] @ ev_sum_mat,axis=0)
    evs_comparison = total_evs_per_year - max_evs_allowed_arr
    evs_over_count = 0
    for row_idx in range(len(evs_comparison[:,:])):
        for col_idx in range(len(evs_comparison[row_idx,:])):
            if evs_comparison[row_idx,col_idx] > 0:
                evs_over_count += evs_comparison[row_idx,col_idx]
    sold_arr[15] = -indv_types[15]
    buy_cost = np.sum(indv_types*buy_cost_mat)
    insure_cost = np.sum(indv_types*insure_cost_mat)
    maintaine_cost = np.sum(indv_types*maintaine_cost_mat)
    sell_cost = np.sum(sold_arr*sell_cost_mat)
    fitness = buy_cost + insure_cost + maintaine_cost + fuel_cost + sell_cost + 100000000*evs_over_count
    return (fitness,)


@njit
def mate(ind1, ind2):
    # roll size swaps
    #swap_roll_arr = np.random.choice(4, 2, replace=False)
    #swap_roll_1 = swap_roll_arr[0] 
    #swap_roll_2 = swap_roll_arr[1] 
    #ind1[:, :, swap_roll_1*5:swap_roll_1*5+5], ind2[:, :, swap_roll_1*5:swap_roll_1*5+5] = ind2[:, :, swap_roll_1*5:swap_roll_1*5+5].copy(), ind1[:, :, swap_roll_1*5:swap_roll_1*5+5].copy()
    #ind1[:, :, swap_roll_2*5:swap_roll_2*5+5], ind2[:, :, swap_roll_2*5:swap_roll_2*5+5] = ind2[:, :, swap_roll_2*5:swap_roll_2*5+5].copy(), ind1[:, :, swap_roll_2*5:swap_roll_2*5+5].copy()

    swap_roll_1 = np.random.randint(0,4)
    ind1[:, :, swap_roll_1*5:swap_roll_1*5+5], ind2[:, :, swap_roll_1*5:swap_roll_1*5+5] = ind2[:, :, swap_roll_1*5:swap_roll_1*5+5].copy(), ind1[:, :, swap_roll_1*5:swap_roll_1*5+5].copy()
    return ind1, ind2


@njit
#UPDATE SIZE RANGE!!!!!!!!!!!!!!!
def mutate(individual, min_veh_req_arr, max_evs_allowed_arr, num_allowable_veh_sold_per_year,type_sum_mat):
    reduced_idv = np.zeros((16,16,12))
    number_sold_mat = get_sold_arr(individual)
    mute_roll = np.random.randint(0,3)
    if mute_roll == 0:
        for depth_idx in range(len(number_sold_mat[:,:,:])):
            reduced_idv[depth_idx] = individual[depth_idx] @ type_sum_mat
        year_roll = np.random.randint(0,16)
        size_roll = np.random.randint(0,4)
        for size_roll in range(size_roll,size_roll+1):
            sold_bought_to_final = number_sold_mat[year_roll:,year_roll,size_roll*3:size_roll*3+3]
            veh_bought_to_final = reduced_idv[year_roll:,year_roll,size_roll*3:size_roll*3+3]
            veh_bought_year = reduced_idv[year_roll,year_roll,size_roll*3:size_roll*3+3]
            avail_veh_idxs = np.nonzero(veh_bought_year > 0)[0]
            if len(avail_veh_idxs) == 0:
                continue
            veh_idx_selected = np.random.choice(avail_veh_idxs)
            num_in_red = -np.sum(sold_bought_to_final[:,veh_idx_selected])
            roll_updated_idx = veh_idx_selected
            while roll_updated_idx == veh_idx_selected:
                roll_updated_idx = np.random.randint(0,3)
            neg_choice_idx = 50
            if num_in_red > 0:
                neg_in_sold = np.nonzero(sold_bought_to_final[:,veh_idx_selected])[0]
                num_veh_total = int(veh_bought_year[veh_idx_selected])
                specific_veh_roll = np.random.randint(0,num_veh_total)
                if specific_veh_roll < num_in_red:
                    neg_choice_idx = np.random.choice(neg_in_sold)
                    veh_bought_to_final[:,veh_idx_selected] = veh_bought_to_final[:,veh_idx_selected] - 1
                    veh_bought_to_final[:,roll_updated_idx] = veh_bought_to_final[:,roll_updated_idx] + 1
                    veh_bought_to_final[neg_choice_idx+1:,roll_updated_idx] = veh_bought_to_final[neg_choice_idx+1:,roll_updated_idx] - 1
                    veh_bought_to_final[neg_choice_idx+1:,veh_idx_selected] = veh_bought_to_final[neg_choice_idx+1:,veh_idx_selected] + 1
                else:
                    veh_bought_to_final[:,veh_idx_selected] = veh_bought_to_final[:,veh_idx_selected] - 1
                    veh_bought_to_final[:,roll_updated_idx] = veh_bought_to_final[:,roll_updated_idx] + 1    
            else: 
                veh_bought_to_final[:,veh_idx_selected] = veh_bought_to_final[:,veh_idx_selected] - 1
                veh_bought_to_final[:,roll_updated_idx] = veh_bought_to_final[:,roll_updated_idx] + 1
            keep_for_n_years_roll = np.random.randint(1,11)
            current_service_years = neg_choice_idx + 1 
            if neg_choice_idx == 50:
                current_service_years = 16 - year_roll
            service_increased = False
            # Increase Veh Service Time
            if current_service_years < keep_for_n_years_roll and current_service_years + year_roll < 16:
                service_increased = True
                idx_to_edit = keep_for_n_years_roll + year_roll - 1
                slices_to_pop = []
                if idx_to_edit>=16:
                    keep_for_n_years_roll = 15 - year_roll + 1
                avail_veh_view = reduced_idv[year_roll+current_service_years,:,size_roll*3:size_roll*3+3]
                veh_nonzero_indices = np.nonzero(avail_veh_view)
                veh_nonzero_pairs = np.column_stack(veh_nonzero_indices)
                veh_selected_row,veh_selected_col = veh_nonzero_pairs[np.random.choice(veh_nonzero_pairs.shape[0])]
                slices_to_pop.append([year_roll+current_service_years,veh_selected_row])
                reduced_idv[year_roll+current_service_years:year_roll+keep_for_n_years_roll, veh_selected_row,size_roll*3 + veh_selected_col] = reduced_idv[year_roll+current_service_years:year_roll+keep_for_n_years_roll,veh_selected_row,size_roll*3 + veh_selected_col] - 1
                reduced_idv[year_roll+current_service_years:year_roll+keep_for_n_years_roll, year_roll, size_roll*3+roll_updated_idx] = reduced_idv[year_roll+current_service_years:year_roll+keep_for_n_years_roll, year_roll, size_roll*3+roll_updated_idx] + 1
                # check for special exception
                if year_roll+keep_for_n_years_roll < 16:
                    if reduced_idv[year_roll+keep_for_n_years_roll-1, veh_selected_row, size_roll*3 + veh_selected_col] < reduced_idv[year_roll+keep_for_n_years_roll, veh_selected_row, size_roll*3 + veh_selected_col]:
                        replacement_buy_roll = np.random.randint(0,3)
                        reduced_idv[year_roll+keep_for_n_years_roll:,year_roll+keep_for_n_years_roll, size_roll*3 + replacement_buy_roll] += 1
                        reduced_idv[year_roll+keep_for_n_years_roll:,veh_selected_row, size_roll*3 + veh_selected_col] -= 1
                        slices_to_pop.append([year_roll+keep_for_n_years_roll,year_roll+keep_for_n_years_roll])
                for view_i,depth_view in enumerate(reduced_idv[year_roll+current_service_years+1:,:,size_roll*3:size_roll*3+3]):
                    negative_positions = np.where(depth_view < 0)
                    if len(negative_positions[0]) > 0:
                        current_depth = year_roll+current_service_years+1+view_i
                        #neg_row = negative_positions[0][0]
                        #neg_col = negative_positions[1][0]
                        pos_positions = np.column_stack(np.where(depth_view > 0))                    
                        #Repair neg index:
                        reduced_idv[current_depth:,veh_selected_row,size_roll*3 + veh_selected_col] = reduced_idv[current_depth:,veh_selected_row,size_roll*3 + veh_selected_col] + 1
                        veh_selected_row,veh_selected_col = pos_positions[np.random.choice(pos_positions.shape[0])]
                        slices_to_pop.append([current_depth,veh_selected_row])
                        reduced_idv[current_depth:,veh_selected_row,size_roll*3 + veh_selected_col] = reduced_idv[current_depth:,veh_selected_row,size_roll*3 + veh_selected_col] - 1
                if len(slices_to_pop) > 0:
                    for slice_info in slices_to_pop:
                        slice_depth = slice_info[0]
                        slice_row = slice_info[1]
                        while slice_depth < 16:
                            reduced_row = reduced_idv[slice_depth,slice_row,size_roll*3:size_roll*3+3]
                            row_evs = reduced_row[0]
                            row_hvo = np.random.randint(0,int(reduced_row[1])+1)
                            row_b20 = reduced_row[1] - row_hvo
                            row_lng = np.random.randint(0,int(reduced_row[2])+1)
                            row_biolng = reduced_row[2] - row_lng
                            poped_row = np.array([row_evs,row_hvo,row_b20,row_lng,row_biolng])
                            individual[slice_depth,slice_row,size_roll*5:size_roll*5+5] = poped_row
                            slice_depth += 1
            # Decrease Veh Service Time
            elif current_service_years > keep_for_n_years_roll and not service_increased:
                idx_to_edit = keep_for_n_years_roll + year_roll
                if idx_to_edit<16:
                    year_dropping = year_roll + keep_for_n_years_roll - 1
                    max_selling_allowed = num_allowable_veh_sold_per_year[year_dropping]
                    current_selling = -np.sum(number_sold_mat[year_dropping,:,:])
                    if max_selling_allowed > current_selling:
                        reduced_idv[year_roll+keep_for_n_years_roll:year_roll+current_service_years,year_roll,size_roll*3+roll_updated_idx] = reduced_idv[year_roll+keep_for_n_years_roll:year_roll+current_service_years,year_roll,size_roll*3+roll_updated_idx] - 1
                        replacement_buy_roll = np.random.randint(0,3)
                        reduced_idv[year_roll + keep_for_n_years_roll:year_roll + current_service_years,year_roll + keep_for_n_years_roll,size_roll*3+replacement_buy_roll] = reduced_idv[year_roll + keep_for_n_years_roll:year_roll + current_service_years,year_roll + keep_for_n_years_roll,size_roll*3+replacement_buy_roll] + 1
                        if replacement_buy_roll == 0:
                            for row_idx,reduced_row in enumerate(reduced_idv[year_roll + keep_for_n_years_roll:year_roll + current_service_years,year_roll + keep_for_n_years_roll,size_roll*3+replacement_buy_roll]):
                                row_evs = reduced_row
                                individual[year_roll + keep_for_n_years_roll+row_idx,year_roll + keep_for_n_years_roll,size_roll*5+replacement_buy_roll] = reduced_row
                        else:
                            offset_map = [0,1,3]
                            for row_idx,reduced_row in enumerate(reduced_idv[year_roll + keep_for_n_years_roll:year_roll + current_service_years,year_roll + keep_for_n_years_roll,size_roll*3+replacement_buy_roll]):
                                offset = offset_map[replacement_buy_roll]
                                row_fa = np.random.randint(0,int(reduced_row)+1)
                                row_fb = int(reduced_row) - row_fa
                                poped_row = np.array([row_fa,row_fb])
                                individual[year_roll + keep_for_n_years_roll+row_idx,year_roll + keep_for_n_years_roll,size_roll*5+offset:size_roll*5+offset+2] = poped_row
            #POP OUT
            for row_idx,reduced_row in enumerate(veh_bought_to_final[:,:]):
                row_evs = reduced_row[0]
                row_hvo = np.random.randint(0,int(reduced_row[1])+1)
                row_b20 = reduced_row[1] - row_hvo
                row_lng = np.random.randint(0,int(reduced_row[2])+1)
                row_biolng = reduced_row[2] - row_lng
                poped_row = np.array([row_evs,row_hvo,row_b20,row_lng,row_biolng])
                individual[year_roll+row_idx,year_roll,size_roll*5:size_roll*5+5] = poped_row

    if mute_roll == 1:        
        #Fuel Mut
        size_roll = np.random.randint(0,4)
        fuel_roll = np.random.randint(0,2)
        fuel_roll_map = [1,3]
        fuel_option_roll = np.random.randint(0,2)
        fuel_option_col = size_roll*5 + fuel_roll_map[fuel_roll] + fuel_option_roll
        alt_map = [1,0]
        fuel_alt_col = size_roll*5 + fuel_roll_map[fuel_roll] + alt_map[fuel_option_roll]
        for depth_idx, depth_slice in enumerate(individual[:,:,:]):
            for row_idx,row_slice in enumerate(individual[depth_idx,:,:]):
                reass_max = int(row_slice[fuel_alt_col])
                if reass_max > 0:
                    reass_roll = np.random.randint(1,reass_max+1)
                    row_slice[fuel_option_col] += reass_roll
                    row_slice[fuel_alt_col] -= reass_roll
    
    if mute_roll == 2:
        #Complete buy reshuffel
        year_roll = np.random.randint(0,16)
        size_roll = np.random.randint(0,4)
        type_roll = np.random.randint(0,3)
        alt_types_map = [[1,2],[0,2],[0,1]]
        #pop_out_fuel_roll = np.random.randint(0,2)
        first_year_slice = reduced_idv[year_roll,year_roll,size_roll*3:size_roll*3+3]
        max_reass = first_year_slice[alt_types_map[type_roll][0]] + first_year_slice[alt_types_map[type_roll][1]]
        #print(reduced_idv[year_roll:,year_roll,size_roll*3:size_roll*3+3])
        if max_reass > 0:
            alt_type_1_current_val = int(first_year_slice[alt_types_map[type_roll][0]])
            alt_type_2_current_val = int(first_year_slice[alt_types_map[type_roll][1]])
            reass_from_alt_1_roll = np.random.randint(0,alt_type_1_current_val+1)
            reass_from_alt_2_roll = np.random.randint(0,alt_type_2_current_val+1)
            reduced_idv[year_roll:,year_roll,size_roll*3+type_roll] += reass_from_alt_1_roll + reass_from_alt_2_roll
            reduced_idv[year_roll:,year_roll,size_roll*3+alt_types_map[type_roll][0]] -= reass_from_alt_1_roll
            reduced_idv[year_roll:,year_roll,size_roll*3+alt_types_map[type_roll][1]] -= reass_from_alt_2_roll
            for row_idx,row in enumerate(reduced_idv[year_roll:,year_roll,size_roll*3:size_roll*3+3]):
                alt_1_value = row[alt_types_map[type_roll][0]]
                if alt_1_value < 0:
                    reduced_idv[year_roll+row_idx:,year_roll,size_roll*3+type_roll] += alt_1_value
                    reduced_idv[year_roll+row_idx:,year_roll,size_roll*3+alt_types_map[type_roll][0]] -= alt_1_value
                alt_2_value = row[alt_types_map[type_roll][1]]
                if alt_2_value < 0:
                    reduced_idv[year_roll+row_idx:,year_roll,size_roll*3+type_roll] += alt_2_value
                    reduced_idv[year_roll+row_idx:,year_roll,size_roll*3+alt_types_map[type_roll][1]] -= alt_2_value
            #print(reduced_idv[year_roll:,year_roll,size_roll*3:size_roll*3+3])
            for row_idx,reduced_row in enumerate(reduced_idv[year_roll:,year_roll,size_roll*3:size_roll*3+3]):
                row_evs = reduced_row[0]
                row_hvo = np.random.randint(0,int(reduced_row[1])+1)
                row_b20 = reduced_row[1] - row_hvo
                row_lng = np.random.randint(0,int(reduced_row[2])+1)
                row_biolng = reduced_row[2] - row_lng
                poped_row = np.array([row_evs,row_hvo,row_b20,row_lng,row_biolng])
                individual[year_roll+row_idx,year_roll,size_roll*5:size_roll*5+5] = poped_row
    return individual,



    

def main(toolbox,min_veh_dict,max_evs_dict,buy_cost_mat,insure_cost_mat,maintaine_cost_mat,fuel_cost_mat,emissions_cost_mat,sell_cost_mat,emissions_constraint_arr,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat):
    #######CONSTRIANT MATS#######
    min_veh_req_arr = np.array([[ 73., 34., 81., 11.],[ 74., 34., 83., 11.],[ 76., 35., 84., 11.],[ 78., 35., 87., 11.],[ 81., 36., 89., 12.],[ 83., 37., 91., 12.],[ 86., 39., 95., 12.],[ 87., 39., 95., 12.],[ 90., 39., 96., 12.],[ 93., 41.,100., 13.],[ 95., 42.,102., 13.],[ 97., 43.,105., 13.],[ 98., 44.,108., 14.],[102., 46.,109., 14.],[104., 47.,112., 14.],[108., 49.,116., 14.],])
    max_evs_allowed_arr = np.array([[9,  10, 30,  1],[9,  10, 31,  1],[9,  11, 31,  1],[38, 25, 67,  8],[40, 25, 69,  9],[41, 26, 71,  9],[81, 37, 91,  11],[82, 37, 91,  11],[84, 37, 92,  11],[93, 41, 100, 13],[95, 42, 102, 13],[97, 43, 105, 13],[98, 44, 108, 14],[102,46, 109, 14],[104,47, 112, 14],[108,49, 116, 14]])
    num_allowable_veh_sold_per_year = np.array([39., 40., 41., 42., 43., 44., 46., 46., 47., 49., 50., 51., 52., 54., 55., 57.])
    type_sum_mat = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]], dtype=np.float64)
    ev_sum_mat = np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]], dtype=np.float64)
    selling_constraint_arr = np.array([39., 40., 41., 42., 43., 44., 46., 46., 47., 49., 50., 51., 52., 54., 55., 57.])
    #######PARAMETERS#######
    #Population Size
    pop_size = 20000
    #Number Generations
    n_gen = 5000
    # Probability of mating
    prob_mating = 0.5
    # Probability of mutating
    prob_mutating = 0.5
    # Number of offspring to produce
    lambda_ = 14000
    #Number of offspring to keep
    mu = 6000

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
        print(f"Generation {gen} best fitness = {min_fitness}")


if __name__ == '__main__':
    # Create toolbox
    toolbox = base.Toolbox()

    # Load in orginal datasets REPLACE WITH COPIES THAT ARE ALREADY IN MEMORY
    vehicles_fuels_df = pd.read_csv("dataset/vehicles_fuels.csv")
    fuels_df = pd.read_csv("dataset/fuels.csv")
    demand_df = pd.read_csv("dataset/demand.csv")
    vehicles_df = pd.read_csv("dataset/vehicles.csv")
    cost_profiles_df = pd.read_csv("dataset/cost_profiles.csv")
    carbon_emissions_df = pd.read_csv("dataset/carbon_emissions.csv")

    range_arr = np.array([102000,102000,102000,102000,102000,106000,106000,106000,106000,106000,118000,118000,118000,118000,118000,73000,73000,730007,73000,73000,])
    max_ranges_per_size = [102000,106000,73000,118000]

    # fuel_opt_demand
    fuel_demand_df = demand_df.copy()
    fuel_demand_df["ID"] = fuel_demand_df["Size"] + "_" + fuel_demand_df["Distance"]
    fuel_demand_df = fuel_demand_df.drop(["Size","Distance"],axis=1)
    fuel_demand_df = fuel_demand_df.pivot(index='Year', columns='ID', values='Demand (km)').reset_index()
    fuel_demand_df = fuel_demand_df.drop(["Year"],axis=1)
    fuel_opt_demand =np.array(fuel_demand_df.values)


    vehicles_cost_df = vehicles_df.copy()
    vehicles_cost_df = vehicles_cost_df.drop(["Yearly range (km)","Distance","Size","Vehicle"], axis=1)
    vehicles_cost_df["Type ID"] = vehicles_cost_df["ID"].apply(lambda x: "_".join(x.split("_")[0:2]))
    vehicles_cost_df = vehicles_cost_df.drop(["ID"], axis=1)
    vehicles_cost_df = vehicles_cost_df.pivot(index='Year', columns='Type ID', values='Cost ($)')
    vehicles_cost_df = vehicles_cost_df.reset_index()
    vehicles_cost_df = vehicles_cost_df[["BEV_S1","Diesel_S1","LNG_S1","BEV_S2","Diesel_S2","LNG_S2","BEV_S3","Diesel_S3","LNG_S3","BEV_S4","Diesel_S4","LNG_S4"]]
    veh_cost_mat = np.array(vehicles_cost_df.values)


    # buy_cost_mat
    buy_cost_mat = np.zeros((16,16,12)).astype(int)
    for depth in range(len(buy_cost_mat[:,1,:])):
        buy_cost_mat[depth,depth,:] = veh_cost_mat[depth,:]


    # insure_cost_mat
    insurance_df = cost_profiles_df[["End of Year","Insurance Cost %"]].copy()
    penalties = []
    for i in range(11,17):
        penalties.append([i,2000])
    pen_add = pd.DataFrame(penalties, columns=["End of Year","Insurance Cost %"])
    insurance_df = pd.concat([insurance_df,pen_add])
    insurance_arr = np.array(insurance_df["Insurance Cost %"].values).astype(int)
    insurance_arr = insurance_arr/100
    insure_cost_mat = np.zeros((16,16,12)).astype(float)
    for depth in range(len(insure_cost_mat)):
        for row in range(len(veh_cost_mat)):
            start_idx = depth
            if row > depth:
                break
            insure_perc = insurance_arr[start_idx-row]
            insure_cost_mat[depth,row,:] = insure_perc*veh_cost_mat[row,:]

    # maintaine_cost_mat
    maintain_df = cost_profiles_df[["End of Year","Maintenance Cost %"]].copy()
    penalties = []
    for i in range(11,17):
        penalties.append([i,2000])
    pen_add = pd.DataFrame(penalties, columns=["End of Year","Maintenance Cost %"])
    maintain_df = pd.concat([maintain_df,pen_add])
    maintain_arr = np.array(maintain_df["Maintenance Cost %"].values)
    maintain_arr = maintain_arr/100
    maintaine_cost_mat = np.zeros((16,16,12)).astype(float)
    for depth in range(len(maintaine_cost_mat)):
        for row in range(len(veh_cost_mat)):
            start_idx = depth
            if row > depth:
                break
            maitain_perc = maintain_arr[start_idx-row]
            maintaine_cost_mat[depth,row,:] = maitain_perc*veh_cost_mat[row,:]

    # sell_cost_mat
    dep_df = cost_profiles_df[["End of Year","Resale Value %"]].copy()
    penalties = []
    for i in range(11,17):
        penalties.append([i,0])
    pen_add = pd.DataFrame(penalties, columns=["End of Year","Resale Value %"])
    dep_df = pd.concat([dep_df,pen_add])
    dep_arr = np.array(dep_df["Resale Value %"].values)
    dep_arr = dep_arr/100
    sell_cost_mat = np.zeros((16,16,12)).astype(float)
    for depth in range(len(sell_cost_mat)):
        for row in range(len(veh_cost_mat)):
            start_idx = depth
            if row > depth:
                break
            dep_perc = dep_arr[start_idx-row]
            sell_cost_mat[depth,row,:] = dep_perc*veh_cost_mat[row,:]

    # emissions_constraint_arr
    emissions_constraint_arr = np.array(carbon_emissions_df["Carbon emission CO2/kg"])


    # min_veh_dict
    min_veh_df = demand_df.copy()
    size_range_df = pd.DataFrame({"Size": ["S1", "S2", "S3", "S4"],"Yearly range (km)": [102000, 106000, 73000, 118000]})

    min_veh_df = pd.merge(min_veh_df,size_range_df,on="Size")

    min_veh_df["Req veh"] = min_veh_df["Demand (km)"]/min_veh_df["Yearly range (km)"]
    min_veh_df["Req veh"] = min_veh_df["Req veh"].apply(lambda x: math.ceil(x))
    min_veh_size_df = min_veh_df.drop(["Distance","Demand (km)","Yearly range (km)"],axis=1)
    min_veh_size_df = min_veh_size_df.groupby(by=["Year", "Size"]).sum().reset_index()
    min_veh_dict = {}
    for record in min_veh_size_df.values:
        if record[0] not in min_veh_dict:
            min_veh_dict[record[0]] = {}
        min_veh_dict[record[0]][record[1]] = record[2]

    # max_evs_dict
    ev_buckets_dict = {2023: ["D1"],2024: ["D1"],2025: ["D1"],2026: ["D1","D2"],2027: ["D1","D2"],2028: ["D1","D2"],2029: ["D1","D2","D3"],2030: ["D1","D2","D3"],2031: ["D1","D2","D3"]}
    max_evs_dict = {}
    for year in ev_buckets_dict:
        buckets = ev_buckets_dict[year]
        year_df =  min_veh_df.drop(["Demand (km)", "Yearly range (km)"], axis=1)
        year_df = year_df[year_df["Year"] == year]
        year_df = year_df[year_df["Distance"].isin(buckets)]
        year_df = year_df.groupby(["Year","Size"]).sum(numeric_only=True).reset_index()
    for year in range(2032,2039):
        max_evs_dict[year] = min_veh_dict[year]


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

    # excess_range_arr
    excess_range_arr = np.zeros_like(fuel_opt_demand)
    for year_idx in range(16):
        for size_idx in range(4):
            bucket_demands = fuel_opt_demand[year_idx,size_idx*4:size_idx*4+4]
            size_range = max_ranges_per_size[size_idx]
            residuals = np.mod(bucket_demands, size_range)
            residuals = np.sort(residuals, axis=0)[::-1]
            excess_range_arr[year_idx,size_idx*4:size_idx*4+4] =  residuals


    main(toolbox,min_veh_dict,max_evs_dict,buy_cost_mat,insure_cost_mat,maintaine_cost_mat,fuel_cost_mat,emissions_cost_mat,sell_cost_mat, emissions_constraint_arr,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat)
    