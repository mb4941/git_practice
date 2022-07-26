# code to solve for strategic monopolist problem (based on "firm_problem_cluster_ab.jl")
# author: Miguel Borrero
# date: 25/07/2022
    


using JuMP  # used for mathematical programming
using Ipopt # solver
using Printf
using LinearAlgebra
using DataFrames
using CSV

path = "/Users/miguelborrero/Desktop/Energy_Transitions/"

# main function. r option refers to ramping costs; original_model option refers to whether include endogenous B in the max rate; linear option refers to whether
# to use logit for for capital aggregation or a linear counterpart.
function solve_firm_problem(yearly_utility::DataFrame, hourly_utility::DataFrame,
  hourly_fuel_utility::DataFrame, parameters::Dict{String,Any}; solver="Ipopt", r = true, original_model = true, linear = true)

  if solver=="Ipopt"
    model = Model(
      optimizer_with_attributes(
        Ipopt.Optimizer, "print_level" => 1, "nlp_scaling_method" => "gradient-based", "max_iter" => 5000));
  elseif solver=="Gurobi"
    model = Model(
      optimizer_with_attributes(
        Gurobi.Optimizer, "NonConvex"=>2, "OutputFlag"=>0, "MIPGap"=>1e-2));
  else
    model = Model(
      optimizer_with_attributes(
        NLopt.Optimizer, "algorithm" => :LN_COBYLA));
  end

  # extracting information from modified "utility_hour_for_Julia"
  T = size(hourly_utility, 1);
  weight = [1/T for i in 1:T];
  @printf("Number of hours is %d\n", T)
  i_int = hourly_utility.import_intercept_with_residual./1000.0;
  i_slope = hourly_utility.import_slope./1000.0;
  l_h = hourly_utility.util_load./1000.0;
  total_load = sum(weight[t]*l_h[t] for t in 1:T);


  # extracting information from "utility_fuel_hourly_for_Julia"
  # what technologies are available?
  G_set = []; 
  mc = [];
  K = [];
  if "COL" in unique(hourly_fuel_utility.modefuel_short)
    push!(G_set, 1);
    push!(mc, hourly_fuel_utility[hourly_fuel_utility.modefuel_short .== "COL",:].MC[1]);
    push!(K, hourly_fuel_utility[hourly_fuel_utility.modefuel_short .== "COL",:].NAMEPLATE[1]);
  else
    push!(mc, 0);
    push!(K, 0);
  end
  if "NG_CC" in unique(hourly_fuel_utility.modefuel_short)
    push!(G_set, 2);
    push!(mc, hourly_fuel_utility[hourly_fuel_utility.modefuel_short .== "NG_CC",:].MC[1]);
    push!(K, hourly_fuel_utility[hourly_fuel_utility.modefuel_short .== "NG_CC",:].NAMEPLATE[1]);
  else
    push!(mc, 0);
    push!(K, 0);
  end
  if "NG_NCC" in  unique(hourly_fuel_utility.modefuel_short)
    push!(G_set, 3);
    push!(mc, hourly_fuel_utility[hourly_fuel_utility.modefuel_short .== "NG_NCC",:].MC[1]);
    push!(K, hourly_fuel_utility[hourly_fuel_utility.modefuel_short .== "NG_NCC",:].NAMEPLATE[1]);
  else
    push!(mc, 0);
    push!(K, 0);
  end
  println(G_set)
  # re escale capacity
  K = K./1000.0
  println(K)

  # equivalent to cardinality of F in notes
  G = size(G_set, 1);
  mc = repeat(transpose(mc), T, 1);
  

  # regulatory parameters
  alpha = parameters["alpha"];
  gamma = parameters["gamma"];
  b = parameters["b"];
  a = 0.1;
  ramp = parameters["ramp"];

  # define objects to solve for
  @variable(model, costs >= 0.0001);
  @variable(model, B >= 0.0001);
  @variable(model, imports[1:T]);
  @variable(model, 0.0 <= eqm_q[1:T,1:G]);

  # capacity constraint on hourly generation for every fuel
  @constraint(model, [t=1:T,g=1:G], eqm_q[t,g] <= K[g]);

  # load variable
  @variable(model, load[1:T]);
  @constraint(model, [t=1:T], imports[t]+sum(eqm_q[t,g] for g in 1:G)==load[t]); 
  @constraint(model, [t=1:T], load[t] == l_h[t]);

  # supply curve
  @variable(model, import_price[1:T]);
  @constraint(model, [t=1:T], imports[t]==i_int[t]+i_slope[t]*import_price[t]);

  # constraint on used and useful capital
  
  if linear

    if !(1 in G_set)
      @constraint(model, B==sum(exp(alpha[g]) * K[g] for g in G_set));
    elseif 1 in G && !(2 in G_set) && !(3 in G_set)
      @NLconstraint(model, B==exp(alpha[1]) * (1.0 + exp(b) * sum(eqm_q[t,1]*weight[t] for t in 1:T)/K[1]) * K[1]);
    else
      @NLconstraint(model, B==exp(alpha[1]) * (1.0 + exp(b) * sum(eqm_q[t,1]*weight[t] for t in 1:T)/K[1]) * K[1]
                              + sum(exp(alpha[g]) * K[g] for g in 2:G));
    end
  else

    if !(1 in G_set)
      @constraint(model, B== sum(exp(alpha[g]) * K[g] for g in G_set));
    elseif 1 in G && !(2 in G_set) && !(3 in G_set)
      @NLconstraint(model, B== exp(alpha[1]) * exp(b*sum(eqm_q[t,1] for t in 1:T)/K[1]) / (1.0 + exp(b*sum(eqm_q[t,1] for t in 1:T)/K[1])) * K[1]);
    else
      @NLconstraint(model, B==  exp(alpha[1]) * exp(b*sum(eqm_q[t,1] for t in 1:T)/K[1]) / (1.0 + exp(b*sum(eqm_q[t,1] for t in 1:T)/K[1])) * K[1]
                                + sum(exp(alpha[g]) * K[g] for g in 2:G));
    end
  end

  # costs (ramp costs are optional but included by default)
  @constraint(model, costs == 8.76*sum(weight[t]*(sum(mc[t,g]*eqm_q[t,g] for g in 1:G) 
  + imports[t]*import_price[t]) for t in 1:T) + ifelse(r, sum(weight[t]*(eqm_q[t,1] - eqm_q[t-1,1])^2 * ramp for t in 2:T), 0));

  # objective function: profits in log
  if original_model
    @NLobjective(model, Max, (1+exp(gamma))*log(B) - exp(gamma)*log(costs));
  else
    @NLobjective(model, Max, exp(gamma)*log(B) - exp(gamma)*log(costs));
  end
  optimize!(model)

  ratio_base = 20.0 #yearly_utility.ratio_base[1]

  profit =  JuMP.value.(B) * (JuMP.value.(costs)/(JuMP.value.(B)*ratio_base))^(-exp(gamma));

  #storing results
  
  results = Dict("status" => @sprintf("%s",
  JuMP.termination_status(model)), 
  "profit" => profit,
  "costs" => JuMP.value.(costs), 
  "eqm_q" => JuMP.value.(eqm_q), 
  "import_q" => JuMP.value.(imports),
  "import_price" => JuMP.value.(import_price),
  "B" => JuMP.value.(B),
  "Tec_n" => G_set,
  "weight" => weight);
  
  

  return results

end


# this part is just for trying out the function


yearly_utility = CSV.read(string(path,"Data/utility_year_for_Julia.csv"), DataFrame);
hourly_utility = CSV.read(string(path,"Data/utility_hour_for_Julia.csv"), DataFrame);
hourly_fuel_utility = CSV.read(string(path,"Data/utility_fuel_hour_for_Julia.csv"), DataFrame);

yearly_utility = filter(:utility_id => id -> id == 195, yearly_utility);
yearly_utility = filter(:est_YEAR => y -> y == 2007, yearly_utility);
hourly_utility = filter(:utility_id => id -> id == 195, hourly_utility);
hourly_utility = filter(:est_YEAR => y -> y == 2007, hourly_utility);
hourly_fuel_utility = filter(:utility_id => id -> id == 195, hourly_fuel_utility);
hourly_fuel_utility = filter(:est_YEAR => y -> y == 2007, hourly_fuel_utility);

#yearly_utility = filter(:utility_id => u-> u!=1307, yearly_utility); 
#hourly_utility = filter(:utility_id => u-> u!=1307, hourly_utility);
#hourly_fuel_utilit = filter(:utility_id => u-> u!=1307, hourly_fuel_utility);  

hourly_utility = filter(:hour_of_year => h -> h <= 2500, hourly_utility);
hourly_fuel_utility = filter(:hour_of_year => h -> h <= 2500, hourly_fuel_utility);


parameters = Dict("alpha" => [0.1, 0.1, 0.1], "gamma" => 0.3, "a" => 0.1, "b" => 0.03, "ramp" => 0.1);

results_firm = solve_firm_problem(yearly_utility, hourly_utility, hourly_fuel_utility, parameters, r = false, linear = false, original_model = true)








