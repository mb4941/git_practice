# code to implement indirect inference estimation
# author: Miguel Borrero
# date: 25/07/2022

using CSV, DataFrames, DataFramesMeta
using JuMP # used for mathematical programming
using Ipopt # solver
using Printf
using LinearAlgebra
using Optim
using Statistics, StatsBase
using BenchmarkTools
using GLM
using FixedEffectModels
using RDatasets
FixedEffectModels.stderror(model::StatisticalModel) = sqrt.(diag(vcov(model)))

path = "/Users/miguelborrero/Desktop/Energy_Transitions/"

include(string(path,"Code/firm_problem_miguel.jl"));

# function to generate the data ready for regressions: both with real data and model data
function GenData(x::Array{Float64,1},yearly_utility::DataFrame, hourly_utility::DataFrame,
    hourly_fuel_utility::DataFrame)

    ## this first section of the function is concerned with generating the data produced by the model ()

    # parameters
    parameters = Dict("alpha" => x[1:3], "gamma" => x[4], "b" => x[5], "ramp" => x[6]); #, "a" => x[7]);

    # initialize annual data frame for regression 2
    fit_results_reg_2 = DataFrame(utility_id = Int64[], est_YEAR = Int64[], modefuel_short = String[], revenue = Float64[], costs = Float64[], utilization = Float64[])

    # initialize hourly data frame for regression 1
    fit_results_reg_1 = DataFrame(utility_id = Int64[], est_YEAR = Int64[], hour_of_year = Int64[], modefuel_short = String[], g_h = Float64[], g_h_lag= Float64[],
     price_fit = Float64[], revenue = Float64[], costs = Float64[])

    # selecting a random sample of weeeks (15 in total)
    weeklist = [sort(sample(1:52, 15, replace = false)) for y in 1:length(unique(yearly_utility.est_YEAR))];
    
    # loop over all utilities
    for k in unique(yearly_utility.utility_id)

        # filter data for the corresponding utility
        yearly_utility_temp = filter(:utility_id => u -> u == k, yearly_utility);
        hourly_utility_temp = filter(:utility_id => u -> u == k, hourly_utility);
        hourly_fuel_utility_temp = filter(:utility_id => u -> u == k, hourly_fuel_utility);

        # calculate ratio base (first equivalent to Mar's formulation)
        yearly_utility_temp.load_sum = combine(groupby(hourly_utility_temp, :est_YEAR), :util_load => sum => :load_sum).load_sum
        yearly_utility_temp.ratio_cost = (yearly_utility_temp.TVGC + yearly_utility_temp.tot_import_cost)./yearly_utility_temp.load_sum;
        yearly_utility_temp = transform(groupby(yearly_utility_temp, :utility_id), :ratio_cost => minimum => :ratio_base);
        yearly_utility_temp.ratio_base = yearly_utility_temp.ratio_base .- 0.01;
        yearly_utility_temp.ratio_base .= ifelse.(yearly_utility_temp.ratio_base .< 0.0, 20.0, yearly_utility_temp.ratio_base);

        # loop over the years
        for y in unique(yearly_utility_temp.est_YEAR)

            # filter the data accordingly
            yearly = filter(:est_YEAR => yr -> yr == y, yearly_utility_temp);
            hourly = filter(:est_YEAR => yr -> yr == y, hourly_utility_temp);
            hourly_fuel = filter(:est_YEAR => yr -> yr == y, hourly_fuel_utility_temp);
            hourly = filter(:Week => w -> w in weeklist[1], hourly)
            hourly_fuel = filter(:Week => w -> w in weeklist[1], hourly_fuel);
            #hourly_fuel = filter(:Week => w -> w in weeklist[y-2005], hourly_fuel);

            # call the optimization function on the filtered data to generate data
            @printf("utility: %d and year: %d\n", k, y)
            results_firm = solve_firm_problem(yearly, hourly, hourly_fuel, parameters, r = false);
            @printf("solved \n")


            if (results_firm["status"] == "LOCALLY_SOLVED") | (results_firm["status"] == "ALMOST_LOCALLY_SOLVED") | (results_firm["status"] == "OPTIMAL")


                # loop over hours to store hourly data on fit_results_reg_1

                function lag(h, f)
                    if h > 1
                        return results_firm["eqm_q"][h-1,f]
                    else
                        return results_firm["eqm_q"][h,f]
                    end
                end

                for h in 1:size(unique(hourly_fuel.hour_of_year))[1]
                    h_year =  unique(hourly_fuel.hour_of_year)[h]
                    temp = replace(results_firm["Tec_n"], 1 => "COL", 2 => "NG_CC", 3 => "NG_NCC")

                    for f in 1:size(results_firm["Tec_n"])[1]
                        push!(fit_results_reg_1, [k, y, h_year, temp[f], results_firm["eqm_q"][h,f], lag(h, f), results_firm["import_price"][h], 
                            results_firm["profit"]-results_firm["costs"], results_firm["costs"]])
                        
                    end
                end
                    
            end

        end

    end

    # aggregate hourly data to produce annual data and store it for regression 2

    fit_results_reg_2 = select(combine(groupby(subset(fit_results_reg_1, :modefuel_short => m -> m .!= "NG_NCC"),["utility_id", "est_YEAR", "modefuel_short"]), :g_h => sum => :g_h_total, :revenue => mean => :revenue, 
        :costs => mean => :costs), [:utility_id, :est_YEAR, :modefuel_short, :revenue, :costs, :g_h_total])
    
    ## this second part of the function will deal with generating the variables and consequently datasets ready for desired regressions

    # REGRESSION 1
    # join generated and actual data based on subset of hours for regression 1
    reg_1_gen = innerjoin(fit_results_reg_1, hourly_fuel_utility[!, [:utility_id, :est_YEAR, :modefuel_short, :hour_of_year, :gload, :NAMEPLATE, :MC, :fuel_priceCOL, :fuel_priceNG, :fuel_price]], 
        on = [:utility_id => :utility_id, :est_YEAR => :est_YEAR, :modefuel_short => :modefuel_short, :hour_of_year => :hour_of_year])

    reg_1_gen = innerjoin(reg_1_gen, hourly_utility[!, [:utility_id, :est_YEAR, :hour_of_year, :util_load]], 
        on = [:utility_id => :utility_id, :est_YEAR => :est_YEAR, :hour_of_year => :hour_of_year])
    
    # create variables and re scale where necessary

    subset!(reg_1_gen, :gload => g -> g .!= 0);
    reg_1_gen[!, "NAMEPLATE"] = reg_1_gen[!, "NAMEPLATE"] ./ 1000.0;
    reg_1_gen[!, "util_load"] = reg_1_gen[!, "util_load"] ./ 1000.0;
    reg_1_gen[!, "gload"] = reg_1_gen[!, "gload"] ./ 1000.0;
    reg_1_gen[!, "g_over_K"] .= reg_1_gen[!, "g_h"] ./ reg_1_gen[!, "NAMEPLATE"];
    reg_1_gen[!, "g_lag_over_K"] .= reg_1_gen[!, "g_h_lag"] ./ reg_1_gen[!, "NAMEPLATE"];
    reg_1_gen[!, "fuel_price_ratio"] .= reg_1_gen[!, "fuel_priceNG"] ./ reg_1_gen[!, "fuel_priceCOL"];
    reg_1_gen[!, "OOD"] .= 0;
    reg_1_gen = @eachrow reg_1_gen begin
                if :price_fit < :MC
                    :OOD = 1
                end
            end
    reg_1_gen[!, "OOD_times_ratio"] .= (reg_1_gen[!, "OOD"] .* reg_1_gen[!, "NAMEPLATE"]) ./ reg_1_gen[!, "gload"];
    reg_1_gen[!, "total_K"] .= transform(groupby(reg_1_gen, ["utility_id", "est_YEAR", "hour_of_year"]), :NAMEPLATE => sum => :total_K )[!, "total_K"];
    reg_1_gen[!, "load_total_K"] .= reg_1_gen[!, "util_load"] ./ reg_1_gen[!, "total_K"];
    reg_1_gen = rename(reg_1_gen, :price_fit => :elec_price)

    # REGRESSION 2

    reg_2_gen = innerjoin(fit_results_reg_2, combine(groupby(reg_1_gen, [:utility_id, :est_YEAR, :modefuel_short]), :gload => sum => :gload, :NAMEPLATE => mean => :NAMEPLATE, :fuel_priceCOL => mean => :fuel_priceCOL, 
    :fuel_priceNG => mean => :fuel_priceNG)[!, [:utility_id, :est_YEAR, :modefuel_short, :gload, :NAMEPLATE, :fuel_priceCOL, :fuel_priceNG]], 
    on = [:utility_id => :utility_id, :est_YEAR => :est_YEAR, :modefuel_short => :modefuel_short])
    tempg1 = unstack(reg_2_gen, [:utility_id, :est_YEAR, :fuel_priceCOL, :fuel_priceNG], :modefuel_short, :g_h_total)

    rename!(tempg1, [:COL, :NG_CC] .=>  [:g_h_total_COL, :g_h_total_NG])
    tempg2 = unstack(reg_2_gen, [:utility_id, :est_YEAR, :fuel_priceCOL, :fuel_priceNG], :modefuel_short, :gload)

    rename!(tempg2, [:COL, :NG_CC] .=>  [:load_total_COL, :load_total_NG])
    tempK = unstack(reg_2_gen, [:utility_id, :est_YEAR, :fuel_priceCOL, :fuel_priceNG], :modefuel_short, :NAMEPLATE)

    rename!(tempK, [:COL, :NG_CC] .=>  [:K_COL, :K_NG])
    mat1 = innerjoin(tempg2, tempg1, on = [:utility_id, :est_YEAR, :fuel_priceCOL, :fuel_priceNG])
    mat2 = innerjoin(mat1, tempK, on = [:utility_id, :est_YEAR, :fuel_priceCOL, :fuel_priceNG])

    mat2[!, "util_revenues"] .= combine(groupby(reg_2_gen, [:utility_id, :est_YEAR]), :revenue => mean => :revenue)[!, :revenue]
    mat2[!, "costs"] .= combine(groupby(reg_2_gen, [:utility_id, :est_YEAR]), :costs => mean => :costs)[!, :costs]
    rename!(mat2, :costs .=> :TVC)
    reg_2_gen = mat2   

    ## this last part of the function will prepare datasets for both regressions involving actual data

    # REGRESSION 1

    reg_1_act = innerjoin(hourly_fuel_utility, hourly_utility[!, [:utility_id, :est_YEAR, :hour_of_year, :util_load, :elec_price]], 
    on = [:utility_id => :utility_id, :est_YEAR => :est_YEAR, :hour_of_year => :hour_of_year])

    subset!(reg_1_act, :gload => g -> g .!= 0);
    reg_1_act[!, "NAMEPLATE"] = reg_1_act[!, "NAMEPLATE"] ./ 1000.0;
    reg_1_act[!, "util_load"] = reg_1_act[!, "util_load"] ./ 1000.0;
    reg_1_act[!, "gload"] = reg_1_act[!, "gload"] ./ 1000.0;
    reg_1_act[!, "g_over_K"] .= reg_1_act[!, "gload"] ./ reg_1_act[!, "NAMEPLATE"];
    reg_1_act[!, "fuel_price_ratio"] .= reg_1_act[!, "fuel_priceNG"] ./ reg_1_act[!, "fuel_priceCOL"];
    reg_1_act[!, "OOD"] .= 0
    reg_1_act = @eachrow reg_1_act begin
                if :elec_price < :MC
                    :OOD = 1
                end
            end
    reg_1_act[!, "OOD_times_ratio"] .= (reg_1_act[!, "OOD"] .* reg_1_act[!, "NAMEPLATE"]) ./ reg_1_act[!, "gload"];
    reg_1_act[!, "total_K"] .= transform(groupby(reg_1_act, ["utility_id", "est_YEAR", "hour_of_year"]), :NAMEPLATE => sum => :total_K )[!, "total_K"];
    reg_1_act[!, "load_total_K"] .= reg_1_act[!, "util_load"] ./ reg_1_act[!, "total_K"];
    reg_1_act[!, "gload_lag"] .= lag(reg_1_act[!, "gload"])
    reg_1_act[!, "gload_lag"] = replace(reg_1_act[!, "gload_lag"], missing => 0.0)
    reg_1_act[!, "g_lag_over_K"] .= reg_1_act[!, "gload_lag"] ./ reg_1_act[!, "NAMEPLATE"]

    # REGRESSION 2

    reg_2_act = combine(groupby(subset(hourly_fuel_utility, :modefuel_short => m -> m .!= "NG_NCC"), [:utility_id, :est_YEAR, :modefuel_short]), :gload => sum => :gload_total, 
    :fuel_priceCOL => mean => :fuel_priceCOL, :fuel_priceNG => mean => :fuel_priceNG, :NAMEPLATE => mean => :NAMEPLATE)
    reg_2_act = innerjoin(reg_2_act, yearly_utility[!, [:utility_id, :est_YEAR, :util_revenues, :TVC]], on = [:utility_id, :est_YEAR])
    templ = unstack(reg_2_act, [:utility_id, :est_YEAR, :util_revenues, :TVC, :fuel_priceCOL, :fuel_priceNG], :modefuel_short, :gload_total)
    rename!(templ, [:COL, :NG_CC] .=>  [:gload_total_COL, :g_load_total_NG])
    tempk = unstack(reg_2_act, [:utility_id, :est_YEAR, :util_revenues, :TVC, :fuel_priceCOL, :fuel_priceNG], :modefuel_short, :NAMEPLATE)
    rename!(tempk, [:COL, :NG_CC] .=>  [:K_COL, :K_NG])
    reg_2_act = innerjoin(templ, tempk, on = [:utility_id, :est_YEAR, :util_revenues, :TVC, :fuel_priceCOL, :fuel_priceNG])


    return reg_1_gen, reg_2_gen, reg_1_act, reg_2_act
end

# function to perform the regressions and return coefficients.
function Reg(data::DataFrame, data2::DataFrame)

    coefs = []
    cov_mat = []

    # REGRESSION 1
    # divide data into fuel type subsets for separate regressions
    
    reg_1_COL = filter(:modefuel_short => m -> m == "COL", data);
    regression_1_COL = reg(reg_1_COL, @formula(g_over_K ~ g_lag_over_K + elec_price + load_total_K + fuel_price_ratio + OOD_times_ratio)) #+ OOD + fe(est_YEAR) + fe(utility_id)))
    append!(coefs, coef(regression_1_COL))
    append!(cov_mat, vcov(regression_1_COL))
    coefs_1_1 = coef(regression_1_COL)
    cov_mat_1_1 = vcov(regression_1_COL)
    reg_1_NG_CC = filter(:modefuel_short => m -> m == "NG_CC", data);
    regression_1_NG_CC = reg(reg_1_NG_CC, @formula(g_over_K ~ g_lag_over_K + elec_price + load_total_K + fuel_price_ratio + OOD_times_ratio)) #+ OOD + fe(est_YEAR) + fe(utility_id)))
    append!(coefs, coef(regression_1_NG_CC))
    append!(cov_mat, vcov(regression_1_NG_CC))
    coefs_1_2 = coef(regression_1_NG_CC)
    cov_mat_1_2 = vcov(regression_1_NG_CC)
    #reg_1_NG_NCC = filter(:modefuel_short => m -> m == "NG_NCC", data);
    #regression_1_NG_NCC = reg(reg_1_NG_NCC, @formula(g_over_K ~ g_lag_over_K + elec_price + load_total_K + fuel_price_ratio + OOD + OOD_times_ratio + fe(est_YEAR) + fe(utility_id)))
    #append!(coefs, coef(regression_1_NG_NCC))
    #append!(cov_mat, vcov(regression_1_NG_NCC))
    #coefs_1_3 = coef(regression_1_NG_NCC)
    #cov_mat_1_3 = vcov(regression_1_NG_NCC)
    

    # REGRESSION 2

    #=
    regression_2 = reg(data2, @formula(revenues ~ fuel_priceCOL + fuel_priceNG + g_h_total_COL + g_h_total_NG + load_total_COL + load_total_NG + K_COL + K_NG + costs))
    append!(coefs, coef(regression_2))
    append!(cov_mat, vcov(regression_2))
    =#


    return [coefs_1_1 coefs_1_2], [[cov_mat_1_1]  [cov_mat_1_2]]

end

# indirect inference function which will take as input the coefficients + var/cov matrix from regressions on data and then will generate the coefficients
# from the generated data and return the GMM value using var/cov as metric. This is the function to be minimized.
function II(x::Array{Float64,1}, yearly_utility::DataFrame, hourly_utility::DataFrame,
    hourly_fuel_utility::DataFrame)

    reg_1_gen, reg_2_gen, reg_1_act, reg_2_act = GenData(x, yearly_utility, hourly_utility, hourly_fuel_utility);
    coefs_act, cov_mat = Reg(reg_1_act, reg_2_act)
    coefs_gen = Reg(reg_1_gen, reg_2_gen)[1]

    
    gmm = 0
    for i in 1:size(coefs_act)[2]
        gmm += transpose(coefs_act[:,i] .- coefs_gen[:,i])*cov_mat[i]*(coefs_act[:,i] .- coefs_gen[:,i])
    end
    

    return gmm
end 

## actual program starts here!

# read data
yearly_utility = CSV.read(string(path,"Data/utility_year_for_Julia.csv"), DataFrame);
hourly_utility = CSV.read(string(path,"Data/utility_hour_for_Julia.csv"), DataFrame);
hourly_fuel_utility = CSV.read(string(path,"Data/utility_fuel_hour_for_Julia.csv"), DataFrame);


yearly_utility = filter(:utility_id => id -> id == 195, yearly_utility);
yearly_utility = filter(:est_YEAR => y -> y in [2006] , yearly_utility);
hourly_utility = filter(:utility_id => id -> id == 195, hourly_utility);
hourly_utility = filter(:est_YEAR => y -> y in [2006], hourly_utility);
hourly_fuel_utility = filter(:utility_id => id -> id == 195, hourly_fuel_utility);
hourly_fuel_utility = filter(:est_YEAR => y -> y in [2006], hourly_fuel_utility);

# filter the data based on NA (only in utility_fuel_hour_for_Julia) in all years and also Mar's cases
#=
yearly_utility = filter(:utility_id => u-> u!=1307, yearly_utility); 
yearly_utility = filter(:utility_id => u-> u!=3046, yearly_utility); 
yearly_utility = filter(:utility_id => u-> u!=3265, yearly_utility); 
filter!(:utility_id => u-> u!=7490, yearly_utility);  # weird ---gap in the middle, it also fails to solve sometimes
filter!(:utility_id => u-> u!=12658, yearly_utility); # only one year
filter!(:utility_id => u-> u!=11249, yearly_utility); # almost always negative profits
filter!(:utility_id => u-> u!=17543, yearly_utility); # almost always negative profits
filter!(:utility_id => u-> u!=18642, yearly_utility); # almost always negative profits
filter!(:utility_id => u -> u in [ 189, 924, 1692, 3046, 4045, 4363, 4716, 5580, 9231, 9234, 9267, 9273, 10005, 11252, 11479, 12647, 12658, 12667, 13100, 13143,
13756, 13781, 13809, 13994, 14077, 14232, 17568, 17633, 17690, 17833, 18315, 19436, 20447, 20847, 20856, 20858, 20860, 21554, 26253, 40211, 40229])
=#


xinit = [0.1, 0.1, 0.1, 0.3, 0.03, 0.03];
II(xinit,yearly_utility, hourly_utility, hourly_fuel_utility)




#r = Optim.optimize(x->evalGMM(x, yearly_utility, hourly_utility, hourly_fuel_utility)[1], xinit, 
  # NelderMead(), Optim.Options(show_trace=false, iterations = 3000))
#exp.(r.minimizer)


