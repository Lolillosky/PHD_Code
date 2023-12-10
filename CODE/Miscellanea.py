import os, shutil
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import ks_2samp
from scipy.stats import rankdata
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.animation as animation
import copy


def delete_content_of_folder(folder):
    '''
    Deletes contents of every element within a folder
    Inputs:
    ------
    folder (str): path of folder
    '''
    if os.path.exists(folder):
        confirm = input(f"The folder '{folder}' exists. Do you want to delete its contents? (y/n)").lower()
        if confirm == "y":
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            print(f"The contents of the folder '{folder}' have been deleted.")
        else:
            print(f"The contents of the folder '{folder}' were not deleted.")
    else:
        print(f"The folder '{folder}' does not exist, but will be created.")
        os.mkdir(folder)




def plot_schocks(number_risk_factors, list_of_shocks, risk_factor_names, bins):

    '''
    Generates plots equivalent to Seaborn's pairplot but 2 distributions can be ploted
    Inputs:
    -------
    * number_risk_factors (int): number of risk factors.
    * schocks_back list(array(num_examples, num_risk_factors)): schocks to be plotted in the back. Must be in np array
    * risk_factor_names (list(str)): list of risk factor names
    * bins (int): number of bins.
    '''

    f, ax = plt.subplots(number_risk_factors,number_risk_factors)

    for i in range(number_risk_factors):
        for j in range(number_risk_factors):
            if (i!=j):
                for s in list_of_shocks:
                    ax[i,j].plot(s[:,i], s[:,j], '.', alpha = 0.5)
            else:
                for s in list_of_shocks:
                    ax[i,j].hist(s[:,i], bins = bins, density = True, alpha = 0.5)
                
            if (j==0):
                ax[i,j].set_ylabel(risk_factor_names[i])
            if (i==5):
                ax[i,j].set_xlabel(risk_factor_names[j])


    f.set_size_inches(15,15)     



def concat_dict_containing_np_arrays(dict_list, axis=0):
    """
    Concatenate numpy arrays from a list of dictionaries along a specified axis.
    
    Parameters:
    - dict_list (list of dict): A list of dictionaries where each dictionary has the same keys 
                                and each key corresponds to a numpy array.
    - axis (int, optional): The axis along which the arrays will be concatenated. Default is 0.
    
    Returns:
    - dict: A dictionary with the same keys as the input dictionaries. Each key corresponds to 
            the concatenated numpy array from all dictionaries in the list.
    
    Raises:
    - ValueError: If dictionaries don't have the same keys or arrays can't be concatenated.
    """

    ret_dict = {}

    if not all(set(d.keys()) == set(dict_list[0].keys()) for d in dict_list):
        raise ValueError("All dictionaries in dict_list must have the same keys.")

    # Loop through the keys in the first dictionary (assuming all dictionaries have the same keys)
    for key in dict_list[0].keys():
        try:
            # For each key, retrieve the corresponding numpy arrays from each dictionary in the list
            list_of_arrays = [x[key] for x in dict_list]

            # Concatenate the arrays along the specified axis and store in the return dictionary
            ret_dict[key] = np.concatenate(list_of_arrays, axis=axis)
        
        except ValueError as e:
            raise ValueError(f"Error concatenating arrays for key '{key}': {str(e)}")

    return ret_dict

import numpy as np
from sklearn.utils import shuffle

def shuffle_arrays_in_dict(my_dict, random_state=None):
    """
    Shuffle arrays in the dictionary based on a common index.
    
    Parameters:
    - my_dict (dict): Dictionary containing arrays to be shuffled.
    - random_state (int, optional): Seed for reproducibility.
    
    Returns:
    - dict: Dictionary with shuffled arrays.
    
    Raises:
    - ValueError: If arrays within the dictionary have mismatched lengths.
    """
    
    # Check if dictionary is not empty
    if not my_dict:
        raise ValueError("The dictionary is empty.")
    
    # Determine the length of the first array in the dictionary for common shuffling
    first_key = list(my_dict.keys())[0]
    length = my_dict[first_key].shape[0]
    
    # Verify that all arrays in the dictionary have the same length
    for key, value in my_dict.items():
        if value.shape[0] != length:
            raise ValueError(f"All arrays in the dictionary must have the same length. Array '{key}' has a mismatched length.")
    
    # Generate a shuffled index based on the length of the arrays
    shuffled_index = shuffle(np.arange(length), random_state=random_state)
    
    # Create a copy of the dictionary and shuffle arrays based on the common index
    my_dict_copy = {}
    for key, value in my_dict.items():
        my_dict_copy[key] = value[shuffled_index]
        
    return my_dict_copy




def plot_plat_charts(hpl, rtpl, fig_ax_list = None, fig_tittle = ''):
    """
    Plots comparison charts for given HPL 
    and RTPL data.
    Both cum probability and rank correlation plots.
    
    Parameters:
    - hpl: List or array of HPL data.
    - rtpl: List or array of RTPL data.
    - fig_tittle: Tittle of the whole figure

    
    Returns:
    - fig: A Matplotlib figure containing two subplots.
    """
    
    if fig_ax_list is None:
        # Create a new figure with two subplots side by side
        fig, ax = plt.subplots(1, 2)
    else:
        fig = fig_ax_list[0]
        ax = fig_ax_list[1]
        
    # First subplot: Cumulative histogram for HPL and RTPL
    ax[0].hist([hpl, rtpl], density=True, histtype="step", cumulative=True, 
               bins=np.unique(np.sort(hpl)), 
               label=['HPL', 'RTPL'])
    ax[0].legend(loc='upper left')
    ax[0].set_xlabel('PL')
    ax[0].set_ylabel('Cumulative Probability')
    
    ax[0].set_title('KS statistic: ' + format(ks_2samp(hpl, rtpl)[0], '0.4f'))

    # Second subplot: Scatter plot of ranked HPL versus RTPL
    ax[1].plot(rankdata(hpl), rankdata(rtpl), '.')
    ax[1].set_xlabel('HPL Rank')
    ax[1].set_ylabel('RTPL Rank')
    
    ax[1].set_title('Rank Correlation:' +  format(spearmanr(hpl, rtpl)[0], '0.4f'))
    # Set figure size
    fig.set_size_inches(13, 5)
    
    fig.suptitle(fig_tittle)
    # Return the figure object
    return fig


def plot_points_predict(simul_results,XY_labels ,save_path_file=None):
    """
    Generates and optionally saves a 3D plot of simulation results.

    This function creates a 3D plot using Plotly, visualizing the data points and
    the model's predictions based on the input simulation results. 

    Parameters:
    simul_results (dict): A dictionary containing 'X1', 'X2', 'Y', and 'model' keys.
                          'X1' and 'X2' are lists or arrays of data points, 'Y' is the 
                          corresponding output, and 'model' is the trained model.
    save_path (str, optional): Path where the HTML output of the plot will be saved and file name. 
                               If not provided, the plot is not saved to a file.

    Returns:
    None
    """

    # Determine the range of X1 and X2 values
    x1_min, x1_max = np.min(simul_results['X1']), np.max(simul_results['X1'])
    x2_min, x2_max = np.min(simul_results['X2']), np.max(simul_results['X2'])

    # Generate a mesh grid for X1 and X2 values
    X1_Grid, X2_Grid = np.meshgrid(np.linspace(x1_min, x1_max, 100), 
                                   np.linspace(x2_min, x2_max, 100))

    # Initialize the grid for model predictions
    Y_Grid = np.zeros((100, 100))

    # Populate Y_Grid with model predictions for each point in the grid
    for i in range(100):
        for j in range(100):
            Y_Grid[i, j] = np.exp(simul_results['model'].score_samples([[X1_Grid[i, j], X2_Grid[i, j]]]))

    # Create lines for the 3D plot
    lines = [go.Scatter3d(x=i, y=j, z=k, mode='lines', line=dict(color='rgb(50, 50, 255)', width=1))
             for i, j, k in zip(X1_Grid, X2_Grid, Y_Grid)]
    lines += [go.Scatter3d(x=i, y=j, z=k, mode='lines', line=dict(color='rgb(50, 50, 255)', width=1))
              for i, j, k in zip(X1_Grid.T, X2_Grid.T, Y_Grid.T)]

    # Add the 3D scatter plot of actual data points
    trace1 = go.Scatter3d(
        x=simul_results['X1'],
        y=simul_results['X2'],
        z=simul_results['Y'],
        mode='markers',
        marker=dict(
            size=2,
            color=simul_results['Y'],  # Color by the Y values
            colorscale='Portland',     # Choose a colorscale
            opacity=1.0
        )
    )
    lines.append(trace1)

    # Define the layout of the plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=XY_labels[0], color='black'),
            yaxis=dict(title=XY_labels[1], color='black'),
            zaxis=dict(title='Density', color='black'),
        ),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10),
        showlegend=False,
    )

    # Generate the figure
    fig = go.Figure(data=lines, layout=layout)
    iplot(fig, filename='elevations-3d-surface')

    # Save the plot as HTML if a save path is provided
    if save_path_file:
        fig.write_html(save_path_file)


def animate_plat_evolution(plat_results, chart_title = "", path_file = None):
    
    # Create a figure for the animation
    fig, ax = plt.subplots(1,2)

    # Animation update function
    def update(frame):
        # Clear the current axes
        ax[0].clear()
        ax[1].clear()


        # Select the data for the current frame
        hpl = plat_results.output_dict['y_true'][frame]
        rtpl = plat_results.output_dict['y_pred'][frame]

        # Call your plot function
        plot_plat_charts(hpl, rtpl,fig_ax_list = [fig, ax],
            fig_tittle=chart_title + f"Plat statistics after minibatch {plat_results.output_dict['batch_count'][frame]}")

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(plat_results.output_dict['y_true']), repeat=True)

    plt.close()

    # return(ani)

    html_output = ani.to_jshtml()

    if path_file is not None:

        with open(path_file, 'w') as file:
            file.write(html_output)

    return html_output

def deep_copy_dict_with_arrays(original_dict):
    # Create a deep copy of the dictionary
    copied_dict = copy.deepcopy(original_dict)

    # Make sure to create deep copies of np.arrays
    for key, value in copied_dict.items():
        if isinstance(value, np.ndarray):
            copied_dict[key] = np.copy(value)

    return copied_dict


      





    












