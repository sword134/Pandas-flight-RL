#from ACUMU.idle_station_optimizer import run_ACUMU #Disabled for stackoverflow to not overcomplicate the problem

def calc_total_reward(df):
    #num_mus = run_ACUMU(filename=df) #Disabled for stackoverflow to not overcomplicate the problem
    num_mus = len(df) #will obviously always return the same, but the issue for Stackoverflow is not the reward calculation but instead the agent space
    cost_mus = 1000*num_mus

    return cost_mus
