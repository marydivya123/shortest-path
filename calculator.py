from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import random

Flask_App = Flask(__name__) # Creating our Flask Instance

@Flask_App.route('/', methods=['GET'])
def index():
    """ Displays the index page accessible at '/' """

    return render_template('index.html')

@Flask_App.route('/operation_result/', methods=['POST'])
def operation_result():
    """Route where we send calculator form input"""
    result = None

    # request.form looks for:
    # html tags with matching "name= "
    first_input = request.form['Input1']
    second_input = request.form['Input2']
    third_input = request.form['Input3']
    operation = request.form['operation']

    input1 = first_input
    input2 = second_input
    input3 = third_input

        # On default, the operation on webpage is addition
    if operation == "+":

            # Setting the parameters gamma and alpha for the Q-Learning
            gamma = 0.75
            alpha = 0.9

            # PART 1 - DEFINING THE ENVIRONMENT
            # Defining the states
            location_to_state = {'A': 0,
                                 'B': 1,
                                 'C': 2,
                                 'D': 3,
                                 'E': 4,
                                 'F': 5,
                                 'G': 6,
                                 'H': 7,
                                 'I': 8,
                                 'J': 9,
                                 'K': 10,
                                 'L': 11}

            # Defining the actions
            actions = [0,1,2,3,4,5,6,7,8,9,10,11]

            # Defining the rewards
            R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
                          [1,0,1,0,0,1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,0,0,1,0,0,0],
                          [0,1,0,0,0,0,0,0,0,1,0,0],
                          [0,0,1,0,0,0,1,1,0,0,0,0],
                          [0,0,0,1,0,0,1,0,0,0,0,1],
                          [0,0,0,0,1,0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0,0,1,0,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,1],
                          [0,0,0,0,0,0,0,1,0,0,1,0]])

            # PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING
            # Making a mapping from the states to the locations
            state_to_location = {state: location for location, state in location_to_state.items()}

            # Making a function that returns the shortest route from a starting to ending location
            def route(starting_location, ending_location):
                R_new = np.copy(R)
                ending_state = location_to_state[ending_location]
                R_new[ending_state, ending_state] = 1000
                Q = np.array(np.zeros([12,12]))
                for i in range(1000):
                    current_state = np.random.randint(0,12)
                    playable_actions = []
                    for j in range(12):
                        if R_new[current_state, j] > 0:
                            playable_actions.append(j)
                    next_state = np.random.choice(playable_actions)
                    TD = R_new[current_state, next_state]+ gamma * Q[next_state, np.argmax(Q[next_state,])]- Q[current_state, next_state]
                    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
                route = [starting_location]
                next_location = starting_location
                while (next_location != ending_location):
                    starting_state = location_to_state[starting_location]
                    next_state = np.argmax(Q[starting_state,])
                    next_location = state_to_location[next_state]
                    route.append(next_location)
                    starting_location = next_location
                return route

            #PART 3 - GOING INTO PRODUCTION
            # Making the final function that returns the optimal route
            def best_route(starting_location, intermediary_location, ending_location):
                return route(starting_location, intermediary_location)+route(intermediary_location, ending_location)[1:]
            # Printing the final route
            print('Route:')
            result=best_route(input1, input2, input3)

    return render_template(
            'index.html',
            input1=input1,
            input2=input2,
            input3=input3,
            operation=operation,
            result=result,
            calculation_success=True
        )


@Flask_App.route('/about/')
def about():
    return render_template('sec.html')

if __name__ == '__main__':
    Flask_App.debug = True
    Flask_App.run()
