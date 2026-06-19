from american_puts import crr_puts_price 

import math 
import numpy as no 

def american_put_greater_than_european():
    args = dict(S0=100 , K=100  , T=1.0 , r=0.05  , sigma=0.25  ,steps=500)
    euro = crr_put_price(**args , american = False)
    amer = crr_put_price(**args , american = True)
    return amer >= euro 


def put_falls_on_spot_rises():
    high_spot = crr_put_price(80,100,1.0,0.05,0.25,500,american=False)
    low_spot = crr_put_price(120,100,1.0,0.05,0.25,500,american=True)

    return low_spot > high_spot


def more_volatility_is_not_cheaper():
    low_vol = crr_put_price(120,100,1.0,0.05,0.15,500,american=True)
    high_vol = crr_put_price(120,100,1.0,0.05,0.35,500,american=True)

    return high_vol >= low_vol


def convergence_table(S0,K,T,r,sigma):
    rows = []

    for steps in [25,50,100,200,500,1000]:
        price = crr_put_price(S0,K,T,r,sigma,steps,american=True)
        rows.append((steps,price))

    return rows


for steps, price in convergence_table(100,100,1.0,0.05,0.25):
    print(f"{steps:4d} steps -> {price:.6f}")