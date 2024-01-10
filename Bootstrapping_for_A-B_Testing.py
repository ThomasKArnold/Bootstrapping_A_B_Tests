###############################################################################################################
# The Lacek Bootstrapping for A/B Testing Python Code
# An A/B bootstrapping program dor testing results from advertising campaigns
###############################################################################################################

###############################################################################################################
# References: The initial code was taken from the following web pages
# https://baotramduong.medium.com/data-science-project-a-b-testing-for-ad-campaign-in-python-ffaca9170bc4
# https://www.datatipz.com/blog/hypothesis-testing-with-bootstrapping-python

###############################################################################################################
# Step1: Load required packages

import streamlit as st
import pandas as pd # For data manipulations
import numpy as np # For numerical calculations
from matplotlib import pyplot as plt # For plotting the distributions
from scipy.stats import norm # For getting the p-values
from scipy.stats import ttest_ind # For getting the t-statistics and p-values
import re

st.title('Bootstrapping for A/B Testing', anchor=None, help=None)
###############################################################################################################
# Step 2: Create Functions
###############################################################################################################

###############################################################################################################
# ttest: Get the t-statistics and p-values

def ttest(df, Group_A, Group_B):
    A_df = df.query("group == '" + Group_A + "'")
    x = np.array(A_df['converted'])
    B_df = df.query("group == '" + Group_B + "'")
    y = np.array(B_df['converted'])

    t_stat, p_val = ttest_ind(y, x)  
    # print("t-statistic = " + str(t_stat))  
    # print("p-value = " + str(p_val))

    return t_stat, p_val

# a, b = ttest(df, 'A', 'B')

###############################################################################################################
# Create a dataframe for a group with n binary values with mean probability = p

def binary_df(group, n, p):
    # column = Name of the binary variable to generate
    # n = Number of total values in the dataframe (Total sample size)
    # p = Mean probability - Make sure n has enough places to cover number of decimal places in probability

    # Create a dataframe with number of ones corresponding to probability
    one_count = int(n*p)
    one_df = pd.DataFrame(columns=['converted'], data=[1] * one_count)

    # Create another dataframe with number of zeroes corresponding to probability
    zero_count = n - one_count
    zero_df = pd.DataFrame(columns=['converted'], data=[0] * zero_count)

    # Combine the ones and zeroes
    df = pd.concat([one_df, zero_df])

    # Shuffle the data by sampling without replacement to make a random looking dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    df['group'] = group
    df = df[['group', 'converted']]

    return df

###############################################################################################################
# Create a dataframe for a group with n binary values with mean probability = p
def test_data_df(groups_df):
    df = pd.DataFrame(columns=['group', 'converted'])

    for i in groups_df.index:
        group = groups_df['group'][i]
        n = groups_df['n'][i]
        p = groups_df['p'][i]
        
        new_df = binary_df(group, n, p)

        if len(df) == 0:
           df = new_df
        else:
           df = pd.concat([df, new_df])

    df = df.reset_index()
    df = df.drop('index', axis=1)
    df = df.reset_index()
    df = df.rename(columns={'index' : 'id'})
    
    return df

###############################################################################################################
# Create a bootstrap sample with n records of the differences in sample means between Group_A and Group_B samples
def bootstrap_sample(df, n, Group_A, Group_B):

    # Bootsrapping
    A_df = df.query("group == '" + Group_A + "'")
    B_df = df.query("group == '" + Group_B + "'")
    # Create a blank list to hold the sample differences
    differences = []
    # Create n bootstrap samples (note that n was set in the settings above)
    for i in range(n):
        # Create a sample dataframe with replacement from group A 
        A_sample_df = A_df.sample(len(A_df), replace=True)
        # Compute the conversion rate for the group A sample
        A_sample_cr = A_sample_df['converted'].mean()

        # Create a sample dataframe with replacement from group B
        B_sample_df = B_df.sample(len(B_df), replace=True)
        # Compute the conversion rate for the group B sample
        B_sample_cr = B_sample_df['converted'].mean()

        # Add the difference between the conversion rates for groups A and B to the differences list
        differences.append(B_sample_cr - A_sample_cr)

    differences = np.array(differences)

    return differences

###############################################################################################################
# Plot the sampling distribution and null hypothesis
def create_the_plot(differences, null_hypothesis, Bootstrap_Mean, Null_Mean, experiment_name, Pair, plot_output_folder):
    plt.figure(figsize=(12, 8))

    plt.hist(differences,  
            alpha=0.5, # the transaparency parameter 
            color='blue',
            bins=25,
            label='Sample Differences') 

    plt.axvline(Bootstrap_Mean, 
                c='blue',
                label='Bootstrap Mean')
    
    plt.hist(null_hypothesis, 
            alpha=0.5, 
            color='red',
            bins=25,
            label='Null Hypothesis') 

    plt.axvline(Null_Mean, 
                c='red',
                label='Null Mean')

    plt.legend(loc='upper right') 

    plot_title = experiment_name + ' (' + Pair + ' Group Pair)'

    plt.title(plot_title)
    plt.xlabel("Difference")
    plt.ylabel("Frequency")

    experiment_name_text = re.sub(r'[\\/*?:"<>|]',"_", experiment_name)
    pair_text = 'groups_' + re.sub(r'[\\/*?:"<>|]',"_", Pair)
    plot_output_file = experiment_name_text + '_' + pair_text + '.png' 
    plot_output_file = plot_output_file.replace(" ", "_")
    plot_output_file = plot_output_folder + experiment_name_text + '_' + pair_text + '.png' 

    # Save the plot to a file
    # plt.savefig(plot_output_file)

    # Show the plot
    plt.show()

###############################################################################################################
# Get the confidence interval for the bootstrap sample
def confidence_interval(differences, bonferoni_alpha):
    # Create a sorted dataframe from the differences
    diffs_df = pd.DataFrame(differences)
    diffs_df.columns = ['difference']
    diffs_df = diffs_df.sort_values(by=['difference'])

    # We want a two sided interval so we compute the size of the sample above and below alpha/2
    # For example for a 95% confidence interval, we have the bottom and top 2.5% of the records (5% / 2)
    conf_int = int(diffs_df.shape[0] * bonferoni_alpha/2)

    # The upper and lower limits are the record value at the lower and upper cuttoff
    # using 1,000 records, this is the 25th and 975th record
    lower = diffs_df['difference'].iloc[conf_int]
    upper = diffs_df['difference'].iloc[diffs_df.shape[0]-conf_int]

    return lower, upper

###############################################################################################################
# Get the p value for the bootstrap sample
def bootstrap_p_value(null_hypothesis, Bootstrap_Mean):
    # Compute and print the bootstrap p_value 
    c_1 = len([x for x in null_hypothesis if x > abs(Bootstrap_Mean)])
    c_2 = len([x for x in null_hypothesis if x < -abs(Bootstrap_Mean)])

    p_value = float((c_1 + c_2)) / float(len(null_hypothesis))

    return p_value

###############################################################################################################
# Get the three star significance
def significance_level(p, bonferoni_alpha_95, bonferoni_alpha_99, bonferoni_alpha_99_9):
    
    if p > bonferoni_alpha_95:
        significance = ''
    elif p > bonferoni_alpha_99:
        significance = '*'
    elif p > bonferoni_alpha_99_9:
        significance = '**'
    else:
        significance = '***'

    return significance

###############################################################################################################
# Finished loading functions
print('Ready to run code')

###############################################################################################################
# Step 3: Set up the environment

# dataset = 'Ford'
# dataset = 'SCJ'
dataset = 'custom'

# Code works with txt or csv files.
# Code assumes three input columns for user id, group name, and conversion measures.
# Code converts your column names to standard column names.

if dataset == 'Ford':
    experiment_name = 'The Ford A/B/C Test (12-28-2023)'
    input_folder = 'Q://TLG/General/Data Solutions/Measure AB Test/'
    input_file_name = 'LARV_Q1_cardholder_test_data_122823.txt'
    text_delimiter = '|'

    # Input column names
    id_column_name = 'consumer_id'
    group_indicator_column_name = 'version'
    conversion_indicator_column_name = 'card_yes'

    n = 10000

    # The plot of the distribution of sample differences vs the plot of the null hypothesis plot will have the following title base
    plot_title_base = 'Distributions of the Sample Differences vs the Null Hypothesis'
    plot_output_folder = 'Q://TLG/General/Data Solutions/Measure AB Test/output/'
elif dataset == 'SCJ':
    experiment_name = 'SCJ Income Test'
    input_folder = 'Q://TLG/General/Data Solutions/Measure AB Test/'
    input_file_name = 'Income_Sample.csv'
    text_delimiter = ','

    # Input column names
    id_column_name = 'Guid'
    group_indicator_column_name = 'YearMonth'
    conversion_indicator_column_name = 'total_excluding_tax'

    n = 10000

    # The plot of the distribution of sample differences vs the plot of the null hypothesis plot will have the following title base
    plot_title_base = 'Distributions of the Sample Differences vs the Null Hypothesis'
    plot_output_folder = 'Q://TLG/General/Data Solutions/Measure AB Test/output/'
elif dataset == 'custom':
    experiment_name = 'Custom A/B/C Test)'
    data = [
        ['A', 10000, .002], 
        ['B', 10000, .0025], 
        ['C', 10000, .001]
    ]

    n = 10000

    plot_title_base = 'Distributions of the Sample Differences vs the Null Hypothesis'
    plot_output_folder = 'Q://TLG/General/Data Solutions/Measure AB Test/output/'

###############################################################################################################
# Step 3a: Housekeeping - No input required

# Start time is set by the program.  No need to edit.
start_time = pd.Timestamp.now()

# Print out the basic overview
print('Results for', experiment_name)

###############################################################################################################
# Step 4: Set up the dataframe

if dataset != 'custom':
    # Put the data into a dataframe.
    # Code works for txt or csv input files, add new type if needed.
    if input_file_name[-3:] == 'txt':
        df = pd.read_table(input_folder + input_file_name, sep=text_delimiter, header='infer')
    elif input_file_name[-3:] == 'csv':
        df = pd.read_csv(input_folder + input_file_name, sep=text_delimiter, header='infer') 

    # Change the data column names to a standard format with “group” and “converted” columns
    df = df.rename(columns={id_column_name : 'id', group_indicator_column_name : 'group', conversion_indicator_column_name : 'converted'})
    df['group'] = df['group'].apply(str)
else:
    columns=['group', 'n', 'p']
    groups_df = pd.DataFrame(data = data, columns = columns)
    df = test_data_df(groups_df)

# df
# df.info()

###############################################################################################################
# Step 5: Get the group stats

group_stats_df = df.groupby('group') \
    .agg({'id':'size', 'converted': ('mean', 'std')}) \
    .reset_index() 
 
group_stats_df.columns = ['Group_ID','N','Conversion_Rate','STD']
print(group_stats_df)

# Number of variables to be tested.
number_of_variables =  group_stats_df.shape[0]
print('Number of Variables:',number_of_variables)
number_of_tests = int((number_of_variables * (number_of_variables - 1)) / 2)
print('Number of Tests:', number_of_tests)

# Calculate the bonferoni alpha for each confidence level
bonferoni_alpha_95 = .05 / number_of_tests
bonferoni_alpha_99 = .01 / number_of_tests
bonferoni_alpha_99_9 = .001 / number_of_tests

# Print out the Bonferoni apha
print('Use {:.4f} as the maximum allowed value for p when interpreting the 95% p-value (alpha = .05).'.format(bonferoni_alpha_95))
print('Use {:.5f} as the maximum allowed value for p when interpreting the 99% p-value (alpha = .01).'.format(bonferoni_alpha_99))
print('Use {:.6f} as the maximum allowed value for p when interpreting the 99.9% p-value (alpha = .001).'.format(bonferoni_alpha_99_9))

# group_stats_df.to_clipboard()
print('Group stats are copied to clipboard')

###############################################################################################################
# STOP HERE to get the group stats for output
###############################################################################################################

###############################################################################################################
# Step 6: Get the test stats

test_df = pd.DataFrame(columns=['Test', 'Pair', 'Group_A', 'Group_B', 'A_N', 'B_N', 'A_CR', 'B_CR', 'A_STD', 'B_STD', 'Diff_B-A', 't', 'p', 't-Significance', 'Boostrap_Mean', 'Lower_95%_CI', 'Upper_95%_CI', 'Lower_99%_CI', 'Upper_99%_CI', 'Lower_99.9%_CI', 'Upper_9.9%_CI','Bootstrap_p', 'Bootstrap_Significance'])

test_counter = 1
for i in range(number_of_tests):
    for j in range(i, number_of_variables-1):

        Group_A = group_stats_df['Group_ID'][i]
        A_N = group_stats_df['N'][i]
        A_CR = group_stats_df['Conversion_Rate'][i]
        A_STD = group_stats_df['STD'][i]
        Group_B = group_stats_df['Group_ID'][j + 1]
        B_N = group_stats_df['N'][j+1]
        B_CR = group_stats_df['Conversion_Rate'][j+1]
        B_STD = group_stats_df['STD'][j+1]
        Diff_B_A = B_CR - A_CR
        Pair = Group_A + '/' + Group_B

        # Get the t and p values
        t, p = ttest(df, Group_A, Group_B)

        t_significance = significance_level(p, bonferoni_alpha_95, bonferoni_alpha_99, bonferoni_alpha_99_9)
        
        # Create the bootstrap sample data
        differences = bootstrap_sample(df, n, Group_A, Group_B)
        Bootstrap_Mean = differences.mean()
        
        # Create a null hypothesis distribution.
        null_hypothesis = np.random.normal(0, differences.std(), differences.size)
        len(null_hypothesis)
        Null_Mean = null_hypothesis.mean()

        # Plot the differences and null_hypothesis distributions.
        create_the_plot(differences, null_hypothesis, Bootstrap_Mean, Null_Mean, experiment_name, Pair, plot_output_folder)

        # Create the confidence intervals
        lower_95, upper_95 = confidence_interval(differences, bonferoni_alpha_95)
        lower_99, upper_99 = confidence_interval(differences, bonferoni_alpha_99)
        lower_99_9, upper_99_9 = confidence_interval(differences, bonferoni_alpha_99_9)

        bootstrap_p = bootstrap_p_value(null_hypothesis, Bootstrap_Mean)

        bootstrap_significance = significance_level(bootstrap_p, bonferoni_alpha_95, bonferoni_alpha_99, bonferoni_alpha_99_9)

        # Put the stats in a list and add a row to the test output dataframe
        column_data = [test_counter, Pair, Group_A, Group_B, A_N, B_N, A_CR, B_CR, A_STD, B_STD, Diff_B_A, t, p, t_significance, \
                    Bootstrap_Mean, lower_95, upper_95, lower_99, upper_99, lower_99_9, upper_99_9, bootstrap_p, bootstrap_significance]
        test_df.loc[len(test_df.index)] = column_data

        test_counter = test_counter + 1

print(test_df)

# test_df.to_clipboard()
print('Test stats are copied to clipboard')

###############################################################################################################
# Show the time to run
end_time = pd.Timestamp.now()
minutes_elapsed = (end_time - start_time)
print('Finished parsing in ' + str(minutes_elapsed))

###############################################################################################################
# The test stats are in the clipboard and can be pasted to Excel
###############################################################################################################



