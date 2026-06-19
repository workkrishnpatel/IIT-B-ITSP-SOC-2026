# core american put function CRR baseline 

import math 
import numpy as np 

def crr_put_price(
        S0: float ,
        K: float,
        T: float,
        r: float,
        sigma: float,
        steps: int,
        american: bool
        ):
    if S0 <= 0 or K <= 0:
        raise ValueError("SO and K must be positive")
    if T <= 0:
        return max(K -S0 , 0.0)
    if sigma <= 0:
        raise ValueError("sigma must be positive ")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be positive inteher")
    
    steps = int(steps)
    dt = T / steps 
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u 
    growth = math.exp(r*dt)
    p = (growth - d) / (u - d)
    disc = math.exp(-r*dt)

    if not (0.0 < p < 1.0):
        raise ValueError("invalid risk neutral probablity : increase the steps size or check your inputs")
    
    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = np.maximum(K - stock , 0.0)

    for i in range(steps - 1 , -1 , -1):
        value = disc * ( p * value[1:i+2] + (1.0 - p) * value[0:i+1])

        if american: 
            j = np.arange(i + 1)
            stock = S0 * (u ** j) * (d ** (i - j))
            exercise = np.maximum(K - stock , 0.0)
            value = np.maximum(value , exercise)

    return float(value[0])       

def crr_put_with_boundary(S0,K,T,r,sigma,steps):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = np.maximum(K - stock,0.0)
    boundary = []

    for i in range(steps - 1,-1,-1):
        continuation = disc * (p * value[1:i + 2] + (1.0 - p) * value[0:i + 1])

        j = np.arange(i + 1)
        stock = S0 * (u ** j) * (d ** (i - j))
        exercise = np.maximum(K - stock,0.0)

        exercise_now = exercise > continuation + 1e-10

        if np.any(exercise_now):
            boundary_stock = float(np.max(stock[exercise_now]))
            boundary.append((i * dt,boundary_stock))

        value = np.maximum(continuation,exercise)

    boundary.reverse()
    return float(value[0]),boundary
    


    
