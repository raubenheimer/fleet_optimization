import numpy as np
from deap import creator
from numba import njit,prange

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
    selling_actual_arr = -np.sum(np.sum(sold_arr,axis=1),axis=1) 
    selling_comparison = selling_actual_arr > selling_constraint_arr
    selling_violations_count = np.sum(selling_comparison) 
    if selling_violations_count > 0:
        fitness = 2760000000.0 + selling_violations_count * 100000000.0
        return fitness
    # CHECK EVS

    #FUEL AND EMISSIONS
    fuel_cost = calc_fuel_cost(individual,fuel_cost_mat,emissions_cost_mat, emissions_constraint_arr,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat)
    if fuel_cost == 3000000000.0:
        return fuel_cost
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
    return fitness

@njit
def mate(ind1, ind2):
    # roll size swaps
    swap_roll_1 = np.random.randint(0,4)
    ind1[:, :, swap_roll_1*5:swap_roll_1*5+5], ind2[:, :, swap_roll_1*5:swap_roll_1*5+5] = ind2[:, :, swap_roll_1*5:swap_roll_1*5+5].copy(), ind1[:, :, swap_roll_1*5:swap_roll_1*5+5].copy()
    return ind1, ind2

@njit
#UPDATE SIZE RANGE!!!!!!!!!!!!!!!
def mutate(individual, num_allowable_veh_sold_per_year, type_sum_mat):
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
    return individual

@njit
def create_pop(pop_size):
    pop = []
    for _ in range(pop_size):
        pop.append(create_individual())
    return pop

@njit(parallel=True)
def evaluate_multiple(pop,buy_cost_mat, insure_cost_mat, maintaine_cost_mat, fuel_cost_mat,sell_cost_mat, emissions_cost_mat, emissions_constraint_arr,type_sum_mat,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat,selling_constraint_arr,max_evs_allowed_arr,ev_sum_mat):
    fitnesses = np.zeros(len(pop))
    for i in prange(len(pop)):
        fitnesses[i] = evaluate(pop[i],buy_cost_mat, insure_cost_mat, maintaine_cost_mat, fuel_cost_mat,sell_cost_mat, emissions_cost_mat, emissions_constraint_arr,type_sum_mat,excess_range_arr,fuel_cost_residual_mat,emissions_residual_mat,selling_constraint_arr,max_evs_allowed_arr,ev_sum_mat)
    return fitnesses

@njit
def varOr(population, lambda_, cxpb, mutpb, num_allowable_veh_sold_per_year, type_sum_mat):
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
            ind1, ind2 = mate(ind1, ind2)
            offspring.append(ind1)
        # Apply mutation    
        elif op_choice < cxpb + mutpb:        
            mutate_idx = np.random.randint(0, pop_size)
            ind = np.copy(population[mutate_idx])
            ind = mutate(ind, num_allowable_veh_sold_per_year, type_sum_mat)
            offspring.append(ind)
        # Apply reproduction    
        else:                
            rep_idx = np.random.choice(pop_size)
            ind = np.copy(population[rep_idx])
            offspring.append(ind)
    return offspring

@njit
def select(pop_fitnesses,mu):
    sorted_indices = np.argsort(pop_fitnesses)    
    return sorted_indices[:mu]


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
    #size_selected = np.random.randint(0,4)
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