import math
import random
from matplotlib import pyplot as plt
import csv
import mpmath

figure, axis = plt.subplots(nrows=1, ncols=2)
figure.suptitle('Curve GA')
plt.ion()
axis[1].set_title("Error evolution")
axis[1].set_ylabel('Error')
axis[1].grid(True)


param_constraints = [
    (500,2000),    # c1 range: 1000 to 1500
    (0.1, 5),    # c2 range: 1e-15 to 1e-13
    (0.01, 0.1)        # c3 range: 0.05 to 0.1
]

def generate_y(c1, c2, c3, ε):
    c2=float(c2)*1e-14
    return c1 * mpmath.exp(c3 * mpmath.log(c2 + ε))

    #return math.sin(c1)+c2*c3

def generate_default_curve_points():
#yhape will have to look into increaments or a way for genetic algorithm to deal with non equal datapoints or not same datapoints
    csv_file_path = r'Disp-Force_ExpRT_ndb50.csv'
    expt_stress = []
    expt_strain = []
    # Read the CSV file
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        
        # Skip the header row if it exists
        next(reader)
        
        # Iterate over each row in the CSV
        for row in reader:
            # Assuming the stress column is at index 0 and strain column is at index 1
            stress = float(row[1])
            strain = float(row[0])
            
            # Append stress and strain values to the respective lists
            expt_stress.append(stress)
            expt_strain.append(strain)

        return expt_strain, expt_stress

def generate_new_curve_points(new_chromosome):
    #new_chromosomes is parameters array
    c1 = new_chromosome[0]
    c2 = new_chromosome[1]
    c3 = new_chromosome[2]
    x_new, y_new = sim.get_xy(c1,c2,c3)
    return x_new,y_new
    
    


def plot_results(x, default_y, new_y, generation, error, current_generation):
    # Clean the curve plot
    axis[0].clear()
    axis[0].grid(True)
    axis[0].set_title("Curve adaptation")

    # Curve adaptation
    axis[0].plot(x, default_y, color='g', label='Default curve')
    axis[0].plot(x, new_y, color='r', label='Generated curve')
    axis[0].legend()
    
    # Error evolution
    axis[1].set_title("Error evolution")
    axis[1].set_ylabel('Error')
    axis[1].grid(True)
    axis[1].set_xlabel(f'Generation #{current_generation}')
    axis[1].plot(generation, error)
    
    
    plt.show()
    plt.pause(0.00001)

def find_best_in_generation(survivors):
    atl = []
    winner = []
    for chromosome in survivors:
        atl.append(chromosome[-1])

    best_in_generation = min(atl)

    for candidate in survivors:
        if best_in_generation == candidate[-1]:
            winner = candidate
    return winner


# Define the parameter constraints


def generate_chromosome(initial_params):
    chromosome = initial_params.copy()  # Use the initial parameter values as the chromosome
    for i, param_range in enumerate(param_constraints):
        lower_bound, upper_bound = param_range
        # Generate random value for the parameter within the specified range
        param_value = random.uniform(lower_bound, upper_bound)
        # Validate the generated value against the constraints
        while param_value < lower_bound or param_value > upper_bound:
            param_value = random.uniform(lower_bound, upper_bound)
        # Add the validated value to the chromosome
        chromosome[i] = param_value
    return chromosome


def reproduction(survivors):
    sons = []
    fathers, mothers = split_list(survivors)
    for chromosome in range(50):
        male_chromosome = fathers[chromosome]
        female_chromosome = mothers[chromosome]
        # Generating 2 sons per couple
        for j in range(2):
            slice_point = random.choice(range(3))

            # Adding additional randomness to the genes mixing
            randomizer = bool(random.getrandbits(1))
            if randomizer:
                subject_a = male_chromosome
                subject_b = female_chromosome
            else:
                subject_a = female_chromosome
                subject_b = male_chromosome
            son = subject_a[:slice_point]
            son.extend(subject_b[slice_point:])
            # print('Generated son #{0}: {1} from fathers: {2} and {3}'.format(chromosome, son, male_chromosome, female_chromosome))
            sons.append(son)
    # Once reproduction has finished we called mutation to generate random mutations
    mutation(sons)
    return sons


def mutation(sons):
    for individual in sons:
        for i in range(3):  # Assuming 3 parameters
            lower_bound, upper_bound = param_constraints[i]
            mutated_value = random.uniform(lower_bound, upper_bound)
            individual[i] = mutated_value



def tournament(fathers):
    # Obtaining default curve values
    default_x, default_y = generate_default_curve_points()
    participants = []
    aptitude_tag_list = []
    winner = []

    number_of_participants = random.choice(range(2, 33))

    for i in range(number_of_participants):
        candidate_chromosome = fathers[random.choice(range(len(fathers)))]
        participants.append(candidate_chromosome)

    # Calculating and appending curve error
    for candidate in participants:
        error = []
        candidate_x, candidate_y = generate_new_curve_points(candidate)
        for point in range(len(default_x)):
            error.append(abs(default_y[point] - candidate_y[point]))
        aptitude_tag = math.sqrt(sum(error))
        aptitude_tag_list.append(aptitude_tag)
        if len(candidate) > 3:
            candidate[-1] = aptitude_tag
        else:
            candidate.append(aptitude_tag)

    # Getting winner chromosome
    winner_value = min(aptitude_tag_list)
    for candidate in participants:
        if winner_value == candidate[-1]:
            winner = candidate

    return winner


def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
