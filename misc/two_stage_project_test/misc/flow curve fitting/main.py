import lib
import csv
# List of dictionaries
FATHERS = []
SONS = []
SURVIVORS = []
# Float list
ERROR = []
# Int list
GENERATION = []

overall_best_params = None
overall_best_error = float('inf')

initial_params=[1300,2.3,0.075]
if __name__ == '__main__':
    # print('Generating default curve points')
    default_x, default_y = lib.generate_default_curve_points()
    # print('Generating new chromosomes')
    best_params_min_error = None  # Initialize with None
    min_error = float('inf')  # Initialize with a large value
    for chromosome in range(5000):
        FATHERS.append(lib.generate_chromosome(initial_params))
    for generation in range(100):
        # print('Generation #{}'.format(generation))
        for tournament in range(500):
            # print('Tournament #{}'.format(tournament))
            SURVIVORS.append(lib.tournament(FATHERS.copy()))

        # print('Plot curve and error from best individual from this generation')
        best_in_generation = lib.find_best_in_generation(SURVIVORS.copy())
        new_x, new_y = lib.generate_new_curve_points(best_in_generation)
        GENERATION.append(generation)
        ERROR.append(best_in_generation[-1])
        
        if best_in_generation[-1] < overall_best_error:
            overall_best_error = best_in_generation[-1]
            overall_best_params = best_in_generation[:-1]

        # Print the current generation's best parameters
        print("======================================================")
        print(f"Generation #{generation} - Best Parameters: {best_in_generation[:-1]}")
        print(best_in_generation[-1])

        # Print the overall best parameters till now
        print(f"Overall Best Parameters: {overall_best_params}")
        
        lib.plot_results(default_x, default_y, new_y, GENERATION, ERROR, generation)

        # print('Substituting individuals')
        SONS = lib.reproduction(SURVIVORS.copy())
        FATHERS.clear()
        FATHERS = SONS.copy()
        SONS.clear()
        SURVIVORS.clear()
