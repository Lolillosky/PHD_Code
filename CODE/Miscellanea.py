import os, shutil
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import ks_2samp
from scipy.stats import rankdata
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.animation as animation
import copy
from plotly.subplots import make_subplots
import datetime
import re
  

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
    
    ax[1].set_title('Rank Correlation: ' +  format(spearmanr(hpl, rtpl)[0], '0.4f'))
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
        hpl = plat_results['y_true'][frame]
        rtpl = plat_results['y_pred'][frame]

        # Call your plot function
        plot_plat_charts(hpl, rtpl,fig_ax_list = [fig, ax],
            fig_tittle=chart_title + f"Plat statistics after minibatch {plat_results['batch_count'][frame]}")

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(plat_results['y_true']), repeat=True)

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


def plot_plat_training(plat_results_list, plat_names_list, path_file_name = None):
    
    f, ax = plt.subplots(1,2, figsize = (10,4))
    
    for plat_result, plat_name in zip(plat_results_list, plat_names_list):
    
        ax[0].plot(plat_result['batch_count'], plat_result['ks_stat'], '.-',label = plat_name)
        ax[1].plot(plat_result['batch_count'], plat_result['rank_corr'], '.-', label = plat_name)
        
    ax[0].legend()
    ax[1].legend()
    
    ax[1].axhline(y = 0.80, color = 'yellow', linestyle = ':')
    ax[1].axhline(y = 0.70, color = 'red', linestyle = ':')
    
    ax[0].axhline(y = 0.09, color = 'yellow', linestyle = ':')
    ax[0].axhline(y = 0.12, color = 'red', linestyle = ':')
    
    if path_file_name is not None:
        
        plt.savefig(path_file_name)
    

def plot_plat_training_plotly(plat_results_list, plat_names_list, path_file_name_html=None,path_file_name_pdf = None):
    # Define a list of colors
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']  # Add more colors if needed

    # Creating subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("KS Statistic", "Rank Correlation"))

    for i, (plat_result, plat_name) in enumerate(zip(plat_results_list, plat_names_list)):
        # Select color from the list
        color = colors[i % len(colors)]

        # KS Statistic plot
        fig.add_trace(
            go.Scatter(x=plat_result['batch_count'], y=plat_result['ks_stat'], 
                       mode='lines', name=plat_name + ' ks', line=dict(color=color, width=2), 
                       marker=dict(size=7)),
            row=1, col=1
        )

        # Rank Correlation plot
        fig.add_trace(
            go.Scatter(x=plat_result['batch_count'], y=plat_result['rank_corr'], 
                       mode='lines', name=plat_name + ' rank corr', line=dict(color=color, width=2), 
                       marker=dict(size=7)),
            row=1, col=2
        )

    # Adding horizontal lines for reference
    fig.add_hline(y=0.80, line_dash="dot", line_color="yellow", row=1, col=2)
    fig.add_hline(y=0.70, line_dash="dot", line_color="red", row=1, col=2)
    fig.add_hline(y=0.09, line_dash="dot", line_color="yellow", row=1, col=1)
    fig.add_hline(y=0.12, line_dash="dot", line_color="red", row=1, col=1)

#     Update layout to resemble Matplotlib
    fig.update_layout(
        height=400, width=1000, 
        plot_bgcolor='white',
        xaxis=dict(showline=True, showgrid=False, gridcolor='lightgrey'),
        yaxis=dict(showline=True, showgrid=False, gridcolor='lightgrey'))
    
    fig.update_yaxes(title_text='Statistic',  # axis label
                 showline=True,  # add line at x=0
                 linecolor='black',  # line color
                 linewidth=2.4, # line size
                 ticks='inside',  # ticks outside axis
                 mirror='allticks',  # add ticks to top/right axes
                 tickwidth=2.4,  # tick width
                 tickcolor='black',  # tick color
                 row=1, col=1)
    
    fig.update_yaxes(showline=True,  # add line at x=0
                 linecolor='black',  # line color
                 linewidth=2.4, # line size
                 ticks='inside',  # ticks outside axis
                 mirror='allticks',  # add ticks to top/right axes
                 tickwidth=2.4,  # tick width
                 tickcolor='black',  # tick color
                 row=1, col=2)
    fig.update_xaxes(title_text='Number of minibatches',
                     showline=True,
                     showticklabels=True,
                     linecolor='black',
                     linewidth=2.4,
                     ticks='inside',
#                      tickfont=font_dict,
                     mirror='allticks',
                     tickwidth=2.4,
                     tickcolor='black',
                     )

    # Show plot
    fig.show()

    # Save plot as HTML if filename provided
    if path_file_name_html:
        fig.write_html(path_file_name_html)
    
    if path_file_name_pdf:
        fig.write_image(path_file_name_pdf)

def get_latest_non_empty_subfolder_and_delete_empty(parent_folder):
    date_regex = re.compile(r'\d{4}-\d{2}-\d{2}')

    # List all subfolders and filter by date format
    folders = [f for f in os.listdir(parent_folder) 
               if os.path.isdir(os.path.join(parent_folder, f)) 
               and date_regex.match(f)]

    # Sort folders by date and find the latest non-empty one
    latest_non_empty_folder = None
    for folder in sorted(folders, key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), reverse=False):
        folder_path = os.path.join(parent_folder, folder)
        if os.listdir(folder_path):
            latest_non_empty_folder = folder
        else:
            # Delete the empty folder
            try:
                os.rmdir(folder_path)
                print(f"Deleted empty folder: {folder}"  )
            except OSError as e:
                print(f"Error deleting {folder}: {e}")

    return latest_non_empty_folder if latest_non_empty_folder else "No non-empty subfolders found."

def check_and_manage_path(path):
    """
    Checks if a specified path exists and manages it based on its content.
    
    - If the path does not exist, it automatically creates it.
    - If the path exists and is empty, does nothing.
    - If the path exists and contains files/directories, it asks the user if they want to delete the content.
    - Forces an error if the user does not approve deletion when prompted.
    
    Parameters:
    path (str): The filesystem path to check and manage.
    
    Raises:
    ValueError: If the user declines deletion of its contents.
    """
    
    # Check if the specified path exists
    if not os.path.exists(path):
        # Create the path automatically if it does not exist
        os.makedirs(path)
        print(f"Path created: {path}")
    else:
        # If the path exists, check if it is empty
        if os.listdir(path):
            # Path is not empty, ask for user approval to delete contents
            delete = input(f"The path {path} is not empty. Do you want to delete its content? (y/n): ").strip().lower()
            if delete == 'y':
                # Delete the directory and its contents, then recreate the directory
                shutil.rmtree(path)
                os.makedirs(path)
                print(f"Content of the path {path} has been deleted.")
            else:
                # Raise an error if the user declines to delete the contents
                raise ValueError("Deletion of the path content was not approved. Exiting.")
        else:
            # If the path is empty, there's nothing to do
            print("The path exists and is empty. Nothing to do.")


def plot_plat_statistics_with_zones(dates, rolling_ks_stat, rolling_rank_corr, path_file_name=None):
#     plt.figure(figsize=(12, 7))

    # Plot the KS statistic and rank correlation with labels
    plt.plot(dates, rolling_ks_stat, label='Rolling KS Statistic', color='blue')
    plt.plot(dates, rolling_rank_corr, label='Rolling Rank Correlation', color='green')

    # Define zone conditions
    green_zone_condition = (rolling_rank_corr > 0.8) & (rolling_ks_stat < 0.09)
    red_zone_condition = (rolling_rank_corr < 0.7) | (rolling_ks_stat > 0.12)

    # Initialize counters for each zone
    green_zone_count = 0
    red_zone_count = 0
    yellow_zone_count = 0

    # Fill background colors based on conditions and count zones
    for i in range(len(dates) - 1):
        if green_zone_condition[i]:
            color = 'green'
            green_zone_count += 1
        elif red_zone_condition[i]:
            color = 'red'
            red_zone_count += 1
        else:
            color = 'yellow'
            yellow_zone_count += 1
        plt.fill_between(dates[i:i + 2], 0, 1, color=color, alpha=0.3, step='pre', edgecolor=None)

    # Add critical level lines with labels
    plt.axhline(y=0.09, color='yellow', linestyle='--')
    plt.axhline(y=0.12, color='red', linestyle='-')
    plt.axhline(y=0.7, color='red', linestyle='-')
    plt.axhline(y=0.8, color='yellow', linestyle='--')

    # Customize the plot
    plt.title('Rolling KS Statistic and Rank Correlation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Statistic Value')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Calculate percentages
    total_days = len(dates) - 1  # Minus one because the loop counts intervals between dates
    green_percent = (green_zone_count / total_days) * 100
    red_percent = (red_zone_count / total_days) * 100
    yellow_percent = (yellow_zone_count / total_days) * 100

    # Display percentages on the plot
    plt.text(0.5, 0.5, f"Green Zone: {green_percent:.2f}%\nYellow Zone: {yellow_percent:.2f}%\nRed Zone: {red_percent:.2f}%", transform=plt.gca().transAxes, verticalalignment='center')

    plt.show()

    if path_file_name is not None:
        plt.savefig(path_file_name)


def compute_rolling_plat_statistic(hpl, rtpl, window):
    
    ks = []
    rc = []
    for i in range(len(hpl) - window):
        
        ks_stat, _ = ks_2samp(hpl[i:i+window], rtpl[i:i+window])
        rank_corr = spearmanr(hpl[i:i+window], rtpl[i:i+window])[0]
        
        ks += [ks_stat]
        rc += [rank_corr]
        
        
    return np.array(ks), np.array(rc)

def calculate_cvar(returns, confidence_level):
    """
    Calculate the Conditional Value at Risk (CVaR) at a specific confidence level.

    Parameters:
    - returns: array-like, the list of returns (or losses) for the investment.
    - confidence_level: float, the confidence level (e.g., 0.95 for 95%).

    Returns:
    - cvar: float, the calculated Conditional Value at Risk.
    """
    # Sort the returns in ascending order
    sorted_returns = np.sort(returns.flatten())

    # Calculate the Value at Risk (VaR) at the given confidence level
    var_index = int(np.ceil((1 - confidence_level) * len(sorted_returns))) - 1
    var = sorted_returns[var_index]

    # Calculate the CVaR as the average of the losses worse than the VaR
    cvar = sorted_returns[:var_index + 1].mean()

    return cvar   

def plat_cvar_scatter(rolling_ks_stat, rolling_rank_corr, cvar, path_file_name=None):
    # Assume the first part of your function remains the same
    
    # New figure for scatter plots
    plt.figure(figsize=(14, 6))
    
    # Convert lists to numpy arrays for efficient element-wise operations
    cvar = np.array(cvar)
    rolling_ks_stat = np.array(rolling_ks_stat)
    rolling_rank_corr = np.array(rolling_rank_corr)
    
    # Define zone conditions
    green_zone_condition = (rolling_rank_corr > 0.8) & (rolling_ks_stat < 0.09)
    red_zone_condition = (rolling_rank_corr < 0.7) | (rolling_ks_stat > 0.12)
    yellow_zone_condition = ~(green_zone_condition | red_zone_condition)  # Not green and not red
    
    # Subplot 1: CVaR vs KS Statistic
    plt.subplot(1, 2, 1)
    plt.scatter(cvar[green_zone_condition], rolling_ks_stat[green_zone_condition], color='green', label='Green Zone')
    plt.scatter(cvar[yellow_zone_condition], rolling_ks_stat[yellow_zone_condition], color='yellow', label='Yellow Zone')
    plt.scatter(cvar[red_zone_condition], rolling_ks_stat[red_zone_condition], color='red', label='Red Zone')
    plt.title('CVaR vs KS Statistic')
    plt.xlabel('CVaR')
    plt.ylabel('KS Statistic')
    plt.legend()
    
    # Subplot 2: CVaR vs Rank Correlation
    plt.subplot(1, 2, 2)
    plt.scatter(cvar[green_zone_condition], rolling_rank_corr[green_zone_condition], color='green', label='Green Zone')
    plt.scatter(cvar[yellow_zone_condition], rolling_rank_corr[yellow_zone_condition], color='yellow', label='Yellow Zone')
    plt.scatter(cvar[red_zone_condition], rolling_rank_corr[red_zone_condition], color='red', label='Red Zone')
    plt.title('CVaR vs Rank Correlation')
    plt.xlabel('CVaR')
    plt.ylabel('Rank Correlation')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Save the figure if a filename is provided
    if path_file_name is not None:
        plt.savefig(path_file_name)
