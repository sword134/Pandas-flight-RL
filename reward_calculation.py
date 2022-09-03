from ACUMU.idle_station_optimizer import run_ACUMU

def calc_total_reward(df):
    num_mus = run_ACUMU(filename=df)

    cost_mus = 1000*num_mus

    return cost_mus