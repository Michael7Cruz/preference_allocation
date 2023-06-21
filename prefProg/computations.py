import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import copy

def AllocateBudget(criteria, preference, PRICES, allocation):
    cost_constraint = copy.deepcopy(PRICES)
    n = len(preference)
    m = len(criteria)
    allocation = np.array(allocation).astype(float)
    # Solve for how many items or alternatives, and subtypes are there
    num_items = 0
    num_subtypes = 0
    
    for i in range(0,len(cost_constraint)):
        for j in range(0,len(cost_constraint[i][1])):
            num_subtypes += 1
            for k in range(0,len(cost_constraint[i][1][j][1])):
                num_items += 1
    
    # sensitivity analysis
    inflation_rate = 6.0 / 100 # 2023
    deviation = 1.0 / 100
    optimal_results = []
    optimal_results_list = []
    num_price_changes = 2
    for q in range(0,num_price_changes):
        if q > 0:
            temp = res.fun*-1
            noise = np.random.normal(inflation_rate,deviation)
            for j in range(0,len(cost_constraint)):
                for k in range(0,len(cost_constraint[j][1])):
                    for l in range(0,len(cost_constraint[j][1][k][1])):
                        cost_constraint[j][1][k][1][l] *= (1+inflation_rate) 

        # inequality constraints matrix
        A = np.zeros([m+1,num_items])
        counter3 = 0
        for h in range(0,m):
            counter = 0
            counter2 = 0
            for i in range(0,len(cost_constraint)):
                for j in range(0,len(cost_constraint[i][1])):
                    for k in range(0,len(cost_constraint[i][1][j][1])):
                        if h != i:
                            A[h,counter] = 0
                        else:
                            A[h,counter] = cost_constraint[i][1][j][1][k]
                        counter += 1
                        A[m,counter2] = cost_constraint[i][1][j][1][k]
                        counter2 += 1
        
        '''
        matrix = []
        matrix2 = []
        matrix3 = np.zeros([num_subtypes,num_items])
        for i in range(0,len(cost_constraint)):
            for j in range(0,len(cost_constraint[i][1])):
                for k in range(0,len(cost_constraint[i][1][j][1])):
                    matrix.append(cost_constraint[i][1][j][1][k])
                matrix2.append(matrix)
                matrix = []
        '''
        """
        counter = 0
        for i in range(0,len(matrix2)):
            for j in range(0,len(matrix2[i])):
                if i == 0:
                    matrix3[i,j] = -1*matrix2[i][j]
                else:
                    matrix3[i,counter] = -1*matrix2[i][j]
                counter += 1
        """
        #A = np.insert(A, m+1, matrix3, axis=0)
        # inequality constraints vector
        b = allocation
        monthly_total = sum(allocation)
        b = np.append(allocation,monthly_total)
        # b = np.insert(b, m+1, -1*np.ones([num_subtypes]), axis=0)

        # coefficients of the linear objective function
        c = np.zeros(num_items)
        counter = 0
        counter2 = 0
        for i in range(0,len(cost_constraint)):
            for j in range(0,len(cost_constraint[i][1])):
                for k in range(0,len(cost_constraint[i][1][j][1])):
                    c[counter2] = -preference[counter]
                    counter2 += 1
                counter += 1

        # solve the problem
        # define allocation boundaries here
        boundaries = [(0,5)]*num_items
        
        if q == 0:
            res = linprog(c, A_ub=A, b_ub=b, bounds=boundaries)
            
            optimal_results.append(res)
            optimal_results_list.append(res.fun*-1)
        elif q > 0:
                res_updated = linprog(c, A_ub=A, b_ub=b, bounds=boundaries)
                optimal_results.append(res_updated)
                optimal_results_list.append(res_updated.fun*-1)
                
    #Plot Result
    #plt.plot(range(0,2),optimal_results_list) 
    #plt.scatter(range(0,2),optimal_results_list, c='grey') 
    #plt.xlabel("Number of Price Increase")
    #plt.ylabel("Maximum Expected Preference")
    #plt.title("Maximum Expected Preference Sensitivity")
    #plt.show() 
    
    optimal_range = abs(optimal_results[0].fun*-1 - optimal_results[num_price_changes-1].fun*-1)/optimal_results[0].fun*-1
    return optimal_range, optimal_results[0]
    
    
    
        
    
              
               
            
        

        


        
    
    
# Uses AHP consistency ratio
def make_weights(criteria):
    # Take the vector size as length of the criteria list
    n = len(criteria)
    
    random_index = {
        1: 0,
        2: 0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45
    }

    if n <= 9: # PAHP
        # Compute the needed eigenvalue while making sure that the consistency ratio is < 0.01
        CR = 0.001
        CI = CR*random_index[n]
        eigval = CI * (n-1) + n
         
        # Generate a random n-dimensional vector
        x = np.random.uniform(0,1,size=(n))
        
        # Normalize the vector to have unit length
        x = x / np.linalg.norm(x)
        
        # Scale the vector by the eigenvalue
        x = eigval * x
      
        # Make the sum of all elements equal to 1
        x = x / sum(x)
        
        # Return the weights as list not numpy array as it cannot be read with Django tags
        return list(x)
    elif n > 9:
        # Make (0,1) direct evaluation vector for the n criteria
        s = np.random.uniform(0,1,size=(n))
        # Make the sum of all elements equal to 1
        x = s / sum(s)
        # Sort the array
        x_sort = np.sort(x)
        # Get the indices after sorting
        x_index = np.argsort(x)
        # Select number of elements
        num_elements = 5
        # Compute the step size
        step_size = n // num_elements
        # Select the indices based on the step_size
        indices = np.arange(0,n,step_size)[:num_elements]
        step_size = len(indices)

        # Select the elements based on the indices
        selected_elements = x_sort[indices]
        # Get the position of the selected elements from the original
        chosen_indices = np.where(np.isin(x_sort, selected_elements))[0]

        # Make AHP based vector for the selected elements making sure of the monotonicity
        # Compute the needed eigenvalue while making sure that the consistency ratio is < 0.01
        CR = 0.0099
        CI = CR*random_index[num_elements]
        eigval = CI * (num_elements-1) + num_elements

        r = np.sort(np.random.rand(num_elements)) # already sorted for monotonicity
        r = r / np.linalg.norm(r)
        r = eigval * r
        r = r / sum(r)
        # Get the indices that were not selected
        not_selected_indices = np.where(~np.isin(np.arange(len(x_sort)), chosen_indices))[0]
        # Solve for the weights in PAHP level using linear interpolation
        # (direct, ahp) at x in x_sort
        x_sort= np.interp(x_sort,selected_elements,r)
        # normalized
        x_sort = x_sort
        #sort back
        x_sort = x_sort[x_index]
        # Return the weights as list not numpy array as it cannot be read with Django tags
        return list(x_sort)
    
def ComputeFinalWeights(criteria, prices, allocation):
    # put the result to the final_weights list
    optimal_range = 100
    optimal_list = []
    final_weights = copy.deepcopy(criteria)
    immanence = [-1]
    
    n = len(final_weights)
    M, Mmax = 0, 100
    N, Nmax = 0, 100 # If Nmax is reached, get allocation with minimum range of max expected preference 
    while N < Nmax: # how much change in maximum preference is allowed within x times price change
        preference_list = []
        immanence_scores = []
        M = 0 # reset
        while M < Mmax:
            
            final_weights = criteria
            
            immanence = []
            # first level
            x = make_weights(criteria)
            for i in range(0,n):
                final_weights[i][0] = x[i] 
            # count the number of elements in the second level

            counter = 0
            for i in range(0,n):
                for j in final_weights[i][1]:
                    counter += 1
            
            y = make_weights(list(np.zeros(counter)))
            
            total2 = 0
            for i in range(0,n):
                sum_yi = 0
                
                for j in range(0,len(final_weights[i][1])):
                    final_weights[i][1][j] = y[j]
                    total2 += final_weights[i][1][j]
                
                for j in range(0,len(final_weights[i][1])):
                    sum_yi += final_weights[i][1][j]
                    
                    
                for j in range(0,len(final_weights[i][1])):
                    final_weights[i][1][j] = final_weights[i][1][j] / sum_yi
            
            if counter > 9:
                for i in range(0,n):
                    sum_yi = 0
                    for j in range(0,len(final_weights[i][1])):
                        sum_yi += final_weights[i][1][j]
                        
                    total1 = 0

                    for j in range(0,len(final_weights[i][1])):
                        total1 += final_weights[i][1][j] * sum_yi 

                    immanence.append((total1/total2)/x[i]) 
            else:
                for i in range(0,n):
                    immanence.append(1)
                 
                         
            
            level2_weights = []
            
            for i in range(0,n):
                for j in range(0,len(final_weights[i][1])):
                    final_weights[i][1][j] = final_weights[i][1][j] * x[i]
                    level2_weights.append(final_weights[i][1][j])
                    
            
            immanence_score = abs(1-np.linalg.norm(immanence))
            immanence_scores.append(immanence_score)
            preference_list.append([immanence_score, [level2_weights,immanence]])
            #print(M,immanence)
            M += 1
        
        #plt.scatter(range(0,M),immanence_scores, c='grey') 
        #plt.plot(range(0,M),[1]*M)
        #plt.xlabel("Number of Iterations")
        #plt.ylabel("Distance of Immanence Norm from 1")
        #plt.title("Maximum Expected Preference Sensitivity")
        #plt.show() 
        the_immanent = min(preference_list, key=lambda x: x[0])
        optimal_range, optimal_results = AllocateBudget(criteria,level2_weights, prices, allocation)
        optimal_list.append([optimal_range,[Nmax,optimal_results,the_immanent[1][1],the_immanent[1][0]]])
        N += 1
    
    
    the_optimal = min(optimal_list, key=lambda x: x[0])
    optimal_range = the_optimal[0]
    optimal_results = the_optimal[1][1]
    print("Number of Iterations:", N)
    print("Immanence Ratios:", np.array(the_optimal[1][2]))
    print("Preference Weights:", np.array(the_optimal[1][3]))
    print()
    print("Range of Maximum Expected Preference:",optimal_range)
    res = optimal_results
    print("Slackness:", optimal_results.slack)
    print('Maximum Expected Preference:',round(res.fun*-1,ndigits = 2),
            '\nAllocation Matrix:', res.x,
            '\nNumber of iterations in optimization performed:', res.nit,
            '\nStatus:', res.message)
        # Your maximum expected preference of the chosen allocation which is about
        # [optimal value] will fall down by [optimal_range]
        # within 10 price changes based on the inflation of 6.1% as of 2023.
    
    #optimality_temp = []
    #for i in optimal_list: 
        #optimality_temp.append(i[0])
    #plt.scatter(range(0,100),optimality_temp, c='grey') 
    #plt.plot(range(0,M),[1]*M)
    #plt.xlabel("Iteration Number")
    #plt.ylabel("Relative Change P0 to P1")
    #plt.title("Maximum Expected Preference Sensitivity")
    #plt.show() 
    
    return final_weights, the_optimal[1][2], level2_weights
    
ComputeFinalWeights([['food', [['Canned Sardines in Tomato Sauce', ['555 Bonus Pack', 'Atami Regular Lid', 'Atami EOC', "Family's Budget Pack Plain", 'Hakone Regular Lid', 'King Cup Regular Lid', 'Lucky 7', 'Mariko Regular Lid', 'Mikado Regular Lid', 'Mikado EOC', 'Saba Phil. Sardines - NCR', 'Saba Phil. Sardines Luz/Viz/Min', 'Toyo Regular Lid', 'Toyo EOC', 'Sallenas Regular Lid', 'Young’s Town Bonus']], ['Condensed milk', ['Jersey Sweetened Condensed Cream']], ['Condensada', ['Alaska', 'Alaska', 'Cow Bell', 'Liberty', 'Liberty']], ['Evaporated Milk', ['Angel Filled Milk']], ['Evaporada', ['Alaska', 'Cow Bell']], ['Powdered Milk', ['Alaska Fortified Powdered Milk', 'Anchor Full Cream Milk', 'Jersey Fortified Instant Powdere', 'Bear Brand', 'Birch Tree Full Cream Milk']], ['Coffee Refill', ['Blend 45 - Supermarket', 'Great Taste (Granules) SMKT', 'Great Taste (Premium) SMKT', 'Nescafe Classic', 'Great Taste (Granules) SMKT', 'Great Taste (Premium) SMKT', 'Nescafe Classic']], ['Coffee 3-in-1 Original', ['Café Puro', 'Great Taste Original Twin Pack', 'Kopiko Black', 'Nescafe Original', 'San Mig Coffee 3-in-1 Original']], ['Bread', ['Pinoy Tasty', 'Pinoy Pandesal (10pcs./pack)']], ['Instant Noodles Chicken and Beef', ['Ho-Mi', 'Lucky Me!', 'Payless', 'Quick Chow']], ['Luncheon Meat', ['CDO Chinese Style', 'Purefoods Chinese Style']], ['Meat Loaf', ['555', 'Argentina', 'CDO', 'Gusto', 'Winner - Supermarket (SMKT)', 'Argentina']], ['Corned Beef', ['Argentina', 'Bingo', 'El Rancho - SMKT', 'Star Corned Beef', 'Winner - SMKT', 'Young’s Town Premium', 'Argentina', 'Ligo Premium - Luzon', 'Ligo Premium - VizMin']], ['Beef Loaf', ['555', 'Argentina', 'Bingo', 'CDO', 'Purefoods', 'El Rancho – SMKT', 'Argentina', 'Purefoods']]]], ['transportation', [['PUJ', ['Student Fare PUJ']]]], ['house_bills', [['rent', ['House Rent']], ['electricity', ['Meralco electricity']]]], ['education', [['school expenses', ['General School Expenses']]]]],
            [['food', [['Canned Sardines in Tomato Sauce', [18.75, 17.77, 18.56, 15.25, 13.4, 18.0, 18.75, 18.0, 17.77, 18.56, 17.25, 17.5, 18.72, 19.58, 18.5, 13.25]], ['Condensed milk', [42.0]], ['Condensada', [33.5, 51.5, 41.0, 34.75, 53.0]], ['Evaporated Milk', [44.0]], ['Evaporada', [31.56, 28.5]], ['Powdered Milk', [44.0, 73.3, 96.25, 50.0, 64.75]], ['Coffee Refill', [18.5, 21.0, 19.75, 21.5, 41.0, 38.5, 43.25]], ['Coffee 3-in-1 Original', [4.7, 8.25, 8.0, 7.25, 6.5]], ['Bread', [40.5, 25.0]], ['Instant Noodles Chicken and Beef', [8.4, 8.75, 7.0, 7.25]], ['Luncheon Meat', [39.0, 33.5]], ['Meat Loaf', [19.5, 23.75, 20.75, 16.5, 18.0, 25.25]], ['Corned Beef', [35.75, 22.0, 28.5, 34.0, 30.75, 34.25, 40.75, 35.8, 36.2]], ['Beef Loaf', [19.5, 22.0, 18.5, 20.75, 18.15, 17.75, 25.0, 24.9]]]], ['transportation', [['PUJ', [12.0]]]], ['house_bills', [['rent', [4000.0]], ['electricity', [2000.0]]]], ['education', [['school expenses', [120.0]]]]],
            [4000,500,8000,500])

# The standard in overestimating and underestimating in immanence test is not given.

