import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf,linewidth=100)
def get_sold_arr(indv):
    num_veh_sold_mat = np.zeros((16,16,12), dtype=np.float64)
    for depth in range(len(indv)-1):
        for row in range(len(indv)):
            num_veh_sold_mat[depth,row,:] = indv[depth+1, row, :] - indv[depth, row, :]
            if(depth == row):
                break
    return num_veh_sold_mat

def indv_checker(individual):
    
    min_veh_req_arr = np.array([[ 73., 34., 81., 11.],[ 74., 34., 83., 11.],[ 76., 35., 84., 11.],[ 78., 35., 87., 11.],[ 81., 36., 89., 12.],[ 83., 37., 91., 12.],[ 86., 39., 95., 12.],[ 87., 39., 95., 12.],[ 90., 39., 96., 12.],[ 93., 41.,100., 13.],[ 95., 42.,102., 13.],[ 97., 43.,105., 13.],[ 98., 44.,108., 14.],[102., 46.,109., 14.],[104., 47.,112., 14.],[108., 49.,116., 14.],])
    size_sum_matrix = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],
                                [0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],
                                [0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1]], dtype=np.float64)
    max_evs_allowed_arr = np.array([[9,  10, 30,  1],[9,  10, 31,  1],[9,  11, 31,  1],[38, 25, 67,  8],[40, 25, 69,  9],[41, 26, 71,  9],[81, 37, 91,  11],[82, 37, 91,  11],[84, 37, 92,  11],[93, 41, 100, 13],[95, 42, 102, 13],[97, 43, 105, 13],[98, 44, 108, 14],[102,46, 109, 14],[104,47, 112, 14],[108,49, 116, 14]])
    ev_sum_matrix = np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,0,0],
                              [0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0],
                              [0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], dtype=np.float64)
    number_sold_mat = get_sold_arr(individual)
    num_allowable_veh_sold_per_year = np.array([39., 40., 41., 42., 43., 44., 46., 46., 47., 49., 50., 51., 52., 54., 55., 57.])
    sold_sum_matrix = np.array([[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]], dtype=np.float64)
    num_veh_sold_per_year = np.zeros_like(num_allowable_veh_sold_per_year)
    num_veh_per_size = np.zeros_like(min_veh_req_arr)
    num_evs_per_size = np.zeros_like(max_evs_allowed_arr)
    
    
    for depth in range(len(individual)):
        num_veh_per_size[depth,:] = np.sum(individual[depth,:,:] @ size_sum_matrix,axis=0)
        num_evs_per_size[depth,:] = np.sum(individual[depth,:,:] @ ev_sum_matrix,axis=0)
        num_veh_sold_per_year[depth] = (np.sum(number_sold_mat[depth,:,:] @ sold_sum_matrix,axis=1)[0])*-1
    
    # CHECK min veh requirment
    if not np.array_equal(min_veh_req_arr,num_veh_per_size):
        raise ValueError("Number of total veh not satisfied")
    # CHECK max evs
    evs_comparison = num_evs_per_size <= max_evs_allowed_arr
    all_years_evs_valid = np.all(evs_comparison)
    if not all_years_evs_valid:
        raise ValueError("Number ev greater than allowed")
    # CHECK fleet sold <= 20%
    sold_comparison = num_veh_sold_per_year <= num_allowable_veh_sold_per_year
    all_years_sold_valid = np.all(sold_comparison)
    if not all_years_sold_valid:
        raise ValueError("Selling constraint not respected")
    # CHECK selling veh pos
    sold_pos = np.any(number_sold_mat.ravel() > 0) #doing it like this for Numba
    if sold_pos:
        raise ValueError("Sold value is pos")
    

def buy_sell_to_csv(indvidual,f_dir):
    # Year | ID | Num_Vehicles | Type | Fuel | Distance_bucket | Distance_per_vehicle(km)
    header = ["Year","ID","Num_Vehicles","Type","Fuel","Distance_bucket","Distance_per_vehicle(km)"]
    bought_list = []
    ids = ["BEV_S1_","Diesel_S1_","LNG_S1_","BEV_S2_","Diesel_S2_","LNG_S2_","BEV_S3_","Diesel_S3_","LNG_S3_","BEV_S4_","Diesel_S4_","LNG_S4_"]
    # Format Buys
    for year_idx, vehs_in_year in enumerate(indvidual):
        year = year_idx + 2023
        vehs_bought = vehs_in_year[year_idx,:]
        for id_idx,veh in enumerate(vehs_bought):
            if veh > 0:
                veh_buy_row = [year,f"{ids[id_idx]}{year}",veh,"Buy","","",0]
                bought_list.append(veh_buy_row)
    buy_df = pd.DataFrame(bought_list,columns=header)

    # Format Sells
    sold_list = []
    sold_arr = get_sold_arr(indvidual)
    for year_idx, sold_in_year in enumerate(sold_arr):
        year = year_idx + 2023
        for sold_year_idx, sold_in_row in enumerate(sold_in_year):
            if sold_year_idx > year_idx:
                break
            for id_idx,sold_veh in enumerate(sold_in_row):
                if sold_veh < 0:
                    veh_sell_row = [year,f"{ids[id_idx]}{sold_year_idx+2023}",-sold_veh,"Sell","","",0]
                    sold_list.append(veh_sell_row)
    sell_df = pd.DataFrame(sold_list,columns=header)
    final_df = pd.concat([buy_df,sell_df])
    final_df.to_csv(f_dir,index=False)

def indv_to_csv(indvidual, f_dir):
    veh_ids = ["BEV_S1_","Diesel_S1_","LNG_S1_","BEV_S2_","Diesel_S2_","LNG_S2_","BEV_S3_","Diesel_S3_","LNG_S3_","BEV_S4_","Diesel_S4_","LNG_S4_"]
    header = ["Current Year", "Veh Bought Year", "ID","Number"]
    df_rows = []
    for year_idx, vehs_in_year in enumerate(indvidual):
            year = year_idx + 2023
            for i,veh_row in enumerate(vehs_in_year):
                veh_year = i + 2023
                if i > year_idx:
                    break
                for j,num_veh in enumerate(veh_row):
                    if num_veh > 0:
                        veh_id = f"{veh_ids[j]}{veh_year}"
                        df_rows.append([year,veh_year,veh_id,num_veh])
    df = pd.DataFrame(df_rows,columns=header)
    df.to_csv(f_dir,index=False)

#indvidual = np.load("lb_opt/lb_2024-06-28 11_55_53.npy_complete.npy")
indvidual = np.load("lb_opt/lb_2024-06-29 15_20_18.npy_complete.npy")

type_sum_mat = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]], dtype=np.float64)
print(indvidual)
indvidual = indvidual @ type_sum_mat
#print(indvidual)
indv_checker(indvidual)
buy_sell_to_csv(indvidual.copy(),"lb_opt/buy_sell_sub.csv")
indv_to_csv(indvidual,"lb_opt\individual.csv")