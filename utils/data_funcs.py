import pandas as pd
import streamlit as st
import numpy as np
import math
from .gen_funcs import get_sold_arr

@st.cache_data
def load_data():
    dtype_dict = {
        'Year': int,
    }
    df1 = pd.read_csv('dataset/carbon_emissions.csv', dtype=dtype_dict)
    df2 = pd.read_csv('dataset/cost_profiles.csv')
    df3 = pd.read_csv('dataset/demand.csv', dtype=dtype_dict)
    df4 = pd.read_csv('dataset/fuels.csv', dtype=dtype_dict)
    df5 = pd.read_csv('dataset/vehicles_fuels.csv')
    df6 = pd.read_csv('dataset/vehicles.csv', dtype=dtype_dict)
    return {
        "Carbon Emissions DataFrame": df1,
        "Cost Profiles DataFrame": df2,
        "Demand DataFrame": df3,
        "Fuels DataFrame": df4,
        "Vehicles Fuels DataFrame": df5,
        "Vehicles DataFrame": df6
    }

@st.cache_data
def calc_cost_mats(dataframes):
    vehicles_fuels_df = dataframes["Vehicles Fuels DataFrame"]
    fuels_df = dataframes["Fuels DataFrame"]
    demand_df = dataframes["Demand DataFrame"]
    vehicles_df = dataframes["Vehicles DataFrame"]
    cost_profiles_df = dataframes["Cost Profiles DataFrame"]
    carbon_emissions_df = dataframes["Carbon Emissions DataFrame"]

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

    return (fuel_opt_demand, buy_cost_mat, insure_cost_mat, maintaine_cost_mat, sell_cost_mat, emissions_constraint_arr, min_veh_dict, max_evs_dict, fuel_cost_mat, fuel_cost_residual_mat, emissions_cost_mat, emissions_residual_mat, excess_range_arr)

def construct_buy_sell_df(indvidual):
    # Year | ID | Num_Vehicles | Type | Fuel | Distance_bucket | Distance_per_vehicle(km)
    indv_types = np.zeros((16,16,12))
    type_sum_mat = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]], dtype=np.float64)
    for depth_idx in range(len(indv_types[:,:,:])):
        indv_types[depth_idx] = indvidual[depth_idx] @ type_sum_mat
    header = ["Year","ID","Num_Vehicles","Type","Fuel","Distance_bucket","Distance_per_vehicle(km)"]
    bought_list = []
    ids = ["BEV_S1_","Diesel_S1_","LNG_S1_","BEV_S2_","Diesel_S2_","LNG_S2_","BEV_S3_","Diesel_S3_","LNG_S3_","BEV_S4_","Diesel_S4_","LNG_S4_"]
    # Format Buys
    for year_idx, vehs_in_year in enumerate(indv_types):
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
    return pd.concat([buy_df,sell_df])
    
@st.cache_data
def construct_fuel_df(fuel_opt_dict):
    fuel_map = ["Electricity","HVO","B20","LNG","BioLNG"]
    headers = ["Year","ID","Num_Vehicles","Type","Fuel","Distance_bucket","Distance_per_vehicle(km)"]
    veh_type_map = ["BEV","Diesel","Diesel","LNG","LNG"]
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
    return pd.DataFrame(rows_list,columns=headers)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
