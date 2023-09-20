import os, shutil
import matplotlib.pyplot as plt

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

    # Example usage:
    # data = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}
    # shuffled_data = shuffle_arrays_in_dict(data)
    # print(shuffled_data)




