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
        print(f"The folder '{folder}' does not exist.")



def plot_schocks(number_risk_factors, schocks_back, schocks_front, risk_factor_names, bins):

    '''
    Generates plots equivalent to Seaborn's pairplot but 2 distributions can be ploted
    Inputs:
    -------
    * number_risk_factors (int): number of risk factors.
    * schocks_back (array(num_examples, num_risk_factors)): schocks to be plotted in the back.
    * schocks_front (array(num_examples, num_risk_factors)): schocks to be plotted in the front.
    * risk_factor_names (list(str)): list of risk factor names
    * bins (int): number of bins.
    '''

    f, ax = plt.subplots(number_risk_factors,number_risk_factors)

    for i in range(number_risk_factors):
        for j in range(number_risk_factors):
            if (i!=j):
                ax[i,j].plot(schocks_back[:,i], schocks_back[:,j], '.', alpha = 0.5)
                ax[i,j].plot(schocks_front.iloc[:,i], schocks_front.iloc[:,j], '.')
            else:
                ax[i,j].hist(schocks_back[:,i], bins = bins, density = True, alpha = 0.5)
                ax[i,j].hist(schocks_front.iloc[:,i], bins = bins, density = True, alpha = 0.5)

            if (j==0):
                ax[i,j].set_ylabel(risk_factor_names[i])
            if (i==5):
                ax[i,j].set_xlabel(risk_factor_names[j])


    f.set_size_inches(15,15)       

