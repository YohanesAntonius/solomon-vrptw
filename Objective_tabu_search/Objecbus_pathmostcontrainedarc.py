from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import numpy as np
import matplotlib.pyplot as plt

def create_data_model():
    data = {}
    data['locations'] = [ 
        (35, 35),
        (41, 49),
        (35, 17),
        (55, 45),
        (55, 20),
        (15, 30),
        (25, 30),
        (20, 50),
        (10, 43),
        (55, 60),
        (30, 60),
        (20, 65),
        (50, 35),
        (30, 25),
        (15, 10),
        (30, 5),
        (10, 20),
        (5, 30),
        (20, 40),
        (15, 60),
        (45, 65),
        (45, 20),
        (45, 10),
        (55, 5), 
        (65, 35),
        (65, 20),
        (45, 30),
        (35, 40),
        (41, 37),
        (64, 42),
        (40, 60),
        (31, 52),
        (35, 69),
        (53, 52),
        (65, 55),
        (63, 65),
        (2, 60),
        (20, 20),
        (5, 5),
        (60, 12),
        (40, 25),
        (42, 7),
        (24, 12),
        (23, 3), 
        (11, 14),
        (6, 38),
        (2, 48),
        (8, 56),
        (13, 52),
        (6, 68),
        (47, 47),
        (49, 58),
        (27, 43),
        (37, 31),
        (57, 29),
        (63, 23),
        (53, 12),
        (32, 12),
        (36, 26),
        (21, 24),
        (17, 34),
        (12, 24),
        (24, 58),
        (27, 69),
        (15, 77),
        (62, 77),
        (49, 73),
        (67, 5),
        (56, 39),
        (37, 47),
        (37, 56),
        (57, 68), 
        (47, 16),
        (44, 17),
        (46, 13),
        (49, 11),
        (49, 42),
        (53, 43),
        (61, 52),
        (57, 48),
        (56, 37),
        (55, 54),
        (15, 47),
        (14, 37),
        (11, 31),
        (16, 22),
        (4, 18),
        (28, 18),
        (26, 52),
        (26, 35),
        (31, 67),
        (15, 19),
        (22, 22),
        (18, 24),
        (26, 27),
        (25, 24),
        (22, 27),
        (25, 21),
        (19, 21),
        (20, 26),
        (18, 18),
    ]
    data['time_windows'] = [ 
        (0, 230),
        (0, 204),
        (0, 202),
        (0, 197),
        (139, 169),
        (0, 199),
        (89, 119),
        (0, 198),
        (85, 115),
        (87, 117),
        (114, 144),
        (57, 87),
        (0, 205),
        (149, 179),
        (32, 62),
        (51, 81),
        (65, 95),
        (147, 177),
        (77, 107),
        (66, 96),
        (116, 146),
        (0, 201),
        (87, 117),
        (58, 88),
        (143, 173),
        (156, 186),
        (0, 208),
        (27, 57),
        (29, 59),
        (53, 83),
        (61, 91),
        (0, 202),
        (131, 161),
        (27, 57),
        (0, 183),
        (133, 163),
        (41, 71),
        (0, 198),
        (73, 103),
        (34, 64),
        (75, 105),
        (87, 117),
        (25, 55),
        (122, 152),
        (59, 89),
        (29, 59),
        (107, 137),
        (41, 71),
        (0, 192),
        (98, 128),
        (0, 203),
        (78, 108),
        (0, 208),
        (85, 115),
        (130, 160),
        (126, 156),
        (120, 150),
        (91, 121),
        (180, 210),
        (0, 202),
        (152, 182),
        (66, 96),
        (48, 78),
        (34, 64),
        (63, 93),
        (49, 79),
        (117, 147),
        (73, 103),
        (132, 162),
        (40, 70),
        (168, 198),
        (67, 97),
        (0, 197),
        (68, 98),
        (139, 169),
        (0, 192),
        (63,  93),
        (169, 199),
        (86, 116),
        (82, 112),
        (168, 198),
        (84, 114),
        (0, 196),
        (0, 198),
        (91, 121),
        (0, 196),
        (84, 114),
        (83, 113),
        (64, 94),
        (166, 196),
        (85, 115),
        (0, 194),
        (18, 48),
        (169, 199),
        (0, 207),
        (0, 205), 
        (0, 204),
        (123, 153),
        (0, 198),
        (73, 103),
        (165, 195)
    ]
    data['demands'] = [0, 10, 7, 13, 19, 26, 3, 5, 9, 16, 16, 12, 19, 23, 20, 8, 19, 2, 12, 17, 9, 11, 18, 29, 3, 6, 17, 16, 16, 9, 21, 27, 23, 11, 14, 8, 5, 8, 16, 31, 9, 5, 5, 7, 18, 16, 1, 27, 36, 30, 13, 10, 9, 14, 18, 2, 6, 7, 18, 28, 3, 13, 19, 10, 9, 20, 25, 25, 36, 6, 5, 15, 25, 9, 8, 18, 13, 14, 3, 23, 6, 26, 16, 11, 7, 41, 35, 26, 9, 15, 3, 1, 2, 22, 27, 20, 11, 12, 10, 9, 17] #jumlah permintaan customer
    data['vehicle_capacities'] = [200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200] # didapatkan dari data kapasitas dari setiap kendaraan sejumlah 200 setiap kendaraan
    data['num_vehicles'] = 25 
    data['depot'] = 0 
    return data

def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    total_load = 0
    solutions = {}
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Jalur Kendaraan Ke - {}:\n'.format(vehicle_id)
        vehicle_solution = []
        route_load = 0
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), solution.Min(time_var),
                solution.Max(time_var))

            vehicle_solution.append(manager.IndexToNode(index))
            node_index = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                        solution.Min(time_var),
                                                        solution.Max(time_var))
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
        plan_output += 'Waktu perjalanan : {}min\n'.format(
            solution.Min(time_var))
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)

        plan_output += 'Beban muatan kendaraan: {}\n'.format(route_load)
        plan_output += 'Jumlah keseluruhan waktu perjalanan: {}min\n'.format(total_time)
        print(plan_output)
        solutions[vehicle_id] = vehicle_solution
        total_time += solution.Min(time_var)
        total_load += route_load
    print('total waktu perjalanan kendaraan: {}min'.format(total_time))
    print('total semua muatan perjalanan: {}\n'.format(total_load))

    return solutions

def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    time_matrix = compute_euclidean_distance_matrix(data['locations'])

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_windows']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        10,  # allow waiting time
        100,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][1][1])
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        False,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()        
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.OBJECTIVE_TABU_SEARCH)
    search_parameters.time_limit.FromSeconds(1)
    search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        solution_dict = print_solution(data, manager, routing, solution)

    coordinates = [
        (35, 35),
        (41, 49),
        (35, 17),
        (55, 45),
        (55, 20),
        (15, 30),
        (25, 30),
        (20, 50),
        (10, 43),
        (55, 60),
        (30, 60),
        (20, 65),
        (50, 35),
        (30, 25),
        (15, 10),
        (30, 5),
        (10, 20),
        (5, 30),
        (20, 40),
        (15, 60),
        (45, 65),
        (45, 20),
        (45, 10),
        (55, 5), #23
        (65, 35),
        (65, 20),
        (45, 30),
        (35, 40),
        (41, 37),
        (64, 42),
        (40, 60),
        (31, 52),
        (35, 69),
        (53, 52),
        (65, 55),
        (63, 65),
        (2, 60),
        (20, 20),
        (5, 5),
        (60, 12),
        (40, 25),
        (42, 7),
        (24, 12),
        (23, 3), #43
        (11, 14),
        (6, 38),
        (2, 48),
        (8, 56),
        (13, 52),
        (6, 68),
        (47, 47),
        (49, 58),
        (27, 43),
        (37, 31),
        (57, 29),
        (63, 23),
        (53, 12),
        (32, 12),
        (36, 26),
        (21, 24),
        (17, 34),
        (12, 24),
        (24, 58),
        (27, 69),
        (15, 77),
        (62, 77),
        (49, 73),
        (67, 5),
        (56, 39),
        (37, 47),
        (37, 56),
        (57, 68), #71
        (47, 16),
        (44, 17),
        (46, 13),
        (49, 11),
        (49, 42),
        (53, 43),
        (61, 52),
        (57, 48),
        (56, 37),
        (55, 54),
        (15, 47),
        (14, 37),
        (11, 31),
        (16, 22),
        (4, 18),
        (28, 18),
        (26, 52),
        (26, 35),
        (31, 67),
        (15, 19),
        (22, 22),
        (18, 24),
        (26, 27),
        (25, 24),
        (22, 27),
        (25, 21),
        (19, 21),
        (20, 26),
        (18, 18), #100
    ]
    X = np.array([x[0] for x in coordinates])
    Y = np.array([x[1] for x in coordinates])

    f, ax = plt.subplots(figsize = [8,6])

    ax.plot(X, Y, 'ko', markersize=8)
    ax.plot(X[0], Y[0], 'gX', markersize=30)

    for i, txt in enumerate(coordinates):
        ax.text(X[i], Y[i], f"{i}")

    vehicle_colors = ["g","k","r", "m", "c","b", "y","g","k","r", "m", "c","b", "y","g","k","r", "m", "c","b", "y","g","k","r", "m"] #warna garis
    for vehicle in solution_dict:
        ax.plot(X[solution_dict[vehicle] + [0]], Y[solution_dict[vehicle] + [0]], f'{vehicle_colors[vehicle]}--')

    ax.set_title("Objective Tabu search - Path most constrained arc")

    plt.show()

if __name__ == '__main__':
    main()
