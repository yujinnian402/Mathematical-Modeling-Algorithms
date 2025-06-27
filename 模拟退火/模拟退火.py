import random
import math

def distance(a,b):
    return math.hypot(a[0] - b[0],a[1] - b[1])

def total_distance(route,cities):
    dist = 0
    for i in range(len(route)):
        dist += distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return dist

def neighbor(route):
    a,b = random.sample(range(len(route)),2)
    route[a],route[b] = route[b],route[a]
    return route

def stimulated_annealing(cities,T_initial = 1000,T_min = 1e-8,alpha=0.995,L=100):
    n=len(cities)
    current_route=list(range(n))
    random.shuffle(current_route)
    current_cost=total_distance(current_route,cities)
    T=T_initial
    best_route=list(current_route)
    best_cost=current_cost

    while T>T_min:
        for _ in range(L):
            new_route=neighbor(list(current_route))
            new_cost=total_distance(new_route,cities)
            delta=new_cost-current_cost

            if delta < 0 or random.random() < math.exp(-delta/T):
                current_route = new_route
                current_cost = new_cost
            if current_cost < new_cost:
                best_route=list(current_route)
                best_cost=current_cost

        T *= alpha
    return best_route,best_cost


cities = [(random.uniform(0,100),random.uniform(0,100)) for _ in range(10)]
best_route, best_cost = stimulated_annealing(cities)

print("最优路线:",best_route)
print("最短距离:",best_cost)



