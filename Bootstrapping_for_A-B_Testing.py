###############################################################################################################
# The Bootstrapping for A/B Testing Python Code
# An A/B bootstrapping program for testing results from advertising campaigns
###############################################################################################################

###############################################################################################################
# References: The initial code was taken from the following web pages
# https://baotramduong.medium.com/data-science-project-a-b-testing-for-ad-campaign-in-python-ffaca9170bc4
# https://www.datatipz.com/blog/hypothesis-testing-with-bootstrapping-python

"""
cd /home/tom/Python/Bootstrapping
source .venv/bin/activate
streamlit run Bootstrapping_for_A-B_Testing.py
"""

###############################################################################################################
# Step1: Load required packages

import streamlit as st
import pandas as pd # For data manipulations
import numpy as np # For numerical calculations
from matplotlib import pyplot as plt # For plotting the distributions
from scipy.stats import norm # For getting the p-values
from scipy.stats import ttest_ind # For getting the t-statistics and p-values
import re
import os

print_environment = 'streamlit'
###############################################################################################################
# Step 2: Create Functions
###############################################################################################################
###############################################################################################################
# Use print if local and streamlit if online
def stream_print(print_environment, text_to_print, font_size):
    if print_environment == 'streamlit':
        if font_size == 'title':
            st.title(text_to_print, anchor=None, help=None)
        elif font_size == 'normal':
            st.write(text_to_print)
        
stream_print(print_environment, 'Bootstrapping for A/B Testing', 'title')

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
def bootstrap_sample(df, n, Group_A, Group_B, large_dataset_threshold=50000, large_dataset_sample_size=10000):

    # Bootstrapping
    A_df = df.query("group == '" + Group_A + "'")
    B_df = df.query("group == '" + Group_B + "'")
    print(f'  Bootstrap: Group {Group_A} has {len(A_df):,} rows, Group {Group_B} has {len(B_df):,} rows')

    # For large datasets, sample a fixed number of rows per iteration instead of
    # resampling the full group, which would be extremely slow.
    A_sample_size = len(A_df)
    B_sample_size = len(B_df)

    if len(A_df) > large_dataset_threshold:
        A_sample_size = large_dataset_sample_size
        print(f'  Group {Group_A} is large ({len(A_df):,} rows). Capping each bootstrap resample to {large_dataset_sample_size:,} rows.')
    if len(B_df) > large_dataset_threshold:
        B_sample_size = large_dataset_sample_size
        print(f'  Group {Group_B} is large ({len(B_df):,} rows). Capping each bootstrap resample to {large_dataset_sample_size:,} rows.')

    print(f'  Starting {n:,} bootstrap iterations...')

    # Create a blank list to hold the sample differences
    differences = []
    # Create n bootstrap samples (note that n was set in the settings above)
    for i in range(n):
        # Print progress every 10%
        if i % (n // 10) == 0:
            print(f'  Bootstrap progress: {i}/{n} ({int(i/n*100)}%)')

        # Create a sample dataframe with replacement from group A
        A_sample_df = A_df.sample(A_sample_size, replace=True)
        # Compute the conversion rate for the group A sample
        A_sample_cr = A_sample_df['converted'].mean()

        # Create a sample dataframe with replacement from group B
        B_sample_df = B_df.sample(B_sample_size, replace=True)
        # Compute the conversion rate for the group B sample
        B_sample_cr = B_sample_df['converted'].mean()

        # Add the difference between the conversion rates for groups A and B to the differences list
        differences.append(B_sample_cr - A_sample_cr)

    print(f'  Bootstrap complete: {n}/{n} (100%)')
    differences = np.array(differences)

    return differences

###############################################################################################################
# Plot the sampling distribution and null hypothesis
def create_the_plot(differences, null_hypothesis, Bootstrap_Mean, Null_Mean, experiment_name, Pair, plot_output_folder):
    # Use an explicit Figure object so st.pyplot() renders correctly in Streamlit
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.hist(differences,
            alpha=0.5,  # the transparency parameter
            color='blue',
            bins=25,
            label='Sample Differences')

    ax.axvline(Bootstrap_Mean,
               c='blue',
               label='Bootstrap Mean')

    ax.hist(null_hypothesis,
            alpha=0.5,
            color='red',
            bins=25,
            label='Null Hypothesis')

    ax.axvline(Null_Mean,
               c='red',
               label='Null Mean')

    ax.legend(loc='upper right')

    plot_title = experiment_name + ' (' + Pair + ' Group Pair)'
    ax.set_title(plot_title)
    ax.set_xlabel("Difference")
    ax.set_ylabel("Frequency")

    experiment_name_text = re.sub(r'[\\/*?:"<>|]', "_", experiment_name)
    pair_text = 'groups_' + re.sub(r'[\\/*?:"<>|]', "_", Pair)
    plot_output_file = plot_output_folder + experiment_name_text + '_' + pair_text + '.png'
    plot_output_file = plot_output_file.replace(" ", "_")

    # Save the plot to a file — create output folder if it doesn't exist
    os.makedirs(plot_output_folder, exist_ok=True)
    fig.savefig(plot_output_file)

    # Display the plot in the Streamlit browser
    st.pyplot(fig)
    plt.close(fig)  # Free memory after displaying

    return plot_output_file

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
# Step 3: Set up the environment via Streamlit sidebar inputs

st.sidebar.header('Configuration')

# Auto-detect the folder where this script lives, then build default data/output paths from it.
script_dir = os.path.dirname(os.path.abspath(__file__))

st.sidebar.subheader('Program Location')
base_folder = st.sidebar.text_input(
    'Main Program Folder',
    value=script_dir,
    help='Auto-detected from script location. Edit if needed.'
)
# Ensure base_folder ends with a separator
base_folder = base_folder.rstrip('/\\') + '/'

# Dataset type selector
dataset = st.sidebar.radio(
    'Dataset Type',
    options=['custom', 'file'],
    format_func=lambda x: 'Custom (synthetic data)' if x == 'custom' else 'File (CSV or TXT)'
)

plot_title_base = 'Distributions of the Sample Differences vs the Null Hypothesis'

if dataset == 'file':
    st.sidebar.subheader('Experiment Settings')
    experiment_name = st.sidebar.text_input('Experiment Name', value='My A/B Test')

    # Accept relative (e.g. "data") or absolute paths for data and output folders
    data_folder_input = st.sidebar.text_input(
        'Data Folder',
        value='data',
        help='Relative to main program folder (e.g. "data"), or enter a full path.'
    )
    input_folder = data_folder_input if os.path.isabs(data_folder_input) else os.path.join(base_folder, data_folder_input) + '/'

    input_file_name = st.sidebar.text_input('Input File Name', value='marketing_AB.csv')
    text_delimiter = st.sidebar.selectbox('Delimiter', options=[',', '|', '	'], format_func=lambda x: {',': 'Comma (CSV)', '|': 'Pipe (TXT)', '	': 'Tab'}.get(x, x))

    output_folder_input = st.sidebar.text_input(
        'Output Folder',
        value='output',
        help='Relative to main program folder (e.g. "output"), or enter a full path.'
    )
    plot_output_folder = output_folder_input if os.path.isabs(output_folder_input) else os.path.join(base_folder, output_folder_input) + '/'

    st.sidebar.caption(f'Data: {input_folder}')
    st.sidebar.caption(f'Output: {plot_output_folder}')

    st.sidebar.subheader('Column Names in Your File')
    id_column_name = st.sidebar.text_input('ID Column', value='user id')
    group_indicator_column_name = st.sidebar.text_input('Group Column', value='test group')
    conversion_indicator_column_name = st.sidebar.text_input('Conversion Column', value='converted')

    n = st.sidebar.number_input('Bootstrap Iterations (n)', min_value=100, max_value=50000, value=10000, step=500)

else:  # custom
    st.sidebar.subheader('Custom Synthetic Data')
    experiment_name = st.sidebar.text_input('Experiment Name', value='Custom A/B/C Test')

    output_folder_input = st.sidebar.text_input(
        'Output Folder',
        value='output',
        help='Relative to main program folder (e.g. "output"), or enter a full path.'
    )
    plot_output_folder = output_folder_input if os.path.isabs(output_folder_input) else os.path.join(base_folder, output_folder_input) + '/'

    st.sidebar.caption(f'Output: {plot_output_folder}')

    n = st.sidebar.number_input('Bootstrap Iterations (n)', min_value=100, max_value=50000, value=10000, step=500)

    st.sidebar.markdown('**Groups** — enter one per line as: `GroupName, SampleSize, ConversionRate`')
    custom_groups_text = st.sidebar.text_area(
        'Group Definitions',
        value='A, 10000, 0.002\nB, 10000, 0.0025\nC, 10000, 0.001'
    )

    # Parse the custom group definitions
    data = []
    for line in custom_groups_text.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 3:
            try:
                data.append([parts[0], int(parts[1]), float(parts[2])])
            except ValueError:
                st.sidebar.error(f'Could not parse line: {line}')

run_analysis = st.sidebar.button('▶ Run Analysis', type='primary')


###############################################################################################################
# Step 3a: Run analysis when button is clicked

if not run_analysis:
    st.info('👈 Configure your settings in the sidebar, then click **Run Analysis** to begin.')
else:
    # Start time is set by the program.  No need to edit.
    start_time = pd.Timestamp.now()

    # Print out the basic overview
    stream_print(print_environment, 'Results for ' + experiment_name, 'normal')

    ###############################################################################################################
    # Step 4: Set up the dataframe

    status = st.status('Loading data...', expanded=True)

    if dataset != 'custom':
        status.write(f'📂 Reading file: {input_folder + input_file_name}')
        if input_file_name[-3:] == 'txt':
            df = pd.read_table(input_folder + input_file_name, sep=text_delimiter, header='infer')
        elif input_file_name[-3:] == 'csv':
            df = pd.read_csv(input_folder + input_file_name, sep=text_delimiter, header='infer')

        # Change the data column names to a standard format with "group" and "converted" columns
        df = df.rename(columns={id_column_name : 'id', group_indicator_column_name : 'group', conversion_indicator_column_name : 'converted'})
        df['group'] = df['group'].apply(str)
        status.write(f'✅ Loaded {len(df):,} rows.')
    else:
        status.write('🔧 Generating synthetic data...')
        columns = ['group', 'n', 'p']
        groups_df = pd.DataFrame(data=data, columns=columns)
        df = test_data_df(groups_df)
        status.write(f'✅ Generated {len(df):,} rows.')

    ###############################################################################################################
    # Step 5: Get the group stats

    status.write('📊 Computing group statistics...')

    group_stats_df = df.groupby('group') \
        .agg({'id':'size', 'converted': ('mean', 'std')}) \
        .reset_index()

    group_stats_df.columns = ['Group_ID','N','Conversion_Rate','STD']
    print(group_stats_df)

    # Number of variables to be tested.
    number_of_variables = group_stats_df.shape[0]
    number_of_tests = int((number_of_variables * (number_of_variables - 1)) / 2)

    # Calculate the bonferoni alpha for each confidence level
    bonferoni_alpha_95   = .05  / number_of_tests
    bonferoni_alpha_99   = .01  / number_of_tests
    bonferoni_alpha_99_9 = .001 / number_of_tests

    print('Number of Variables:', number_of_variables)
    print('Number of Tests:', number_of_tests)
    print('Use {:.4f} as the maximum allowed value for p when interpreting the 95% p-value (alpha = .05).'.format(bonferoni_alpha_95))
    print('Use {:.5f} as the maximum allowed value for p when interpreting the 99% p-value (alpha = .01).'.format(bonferoni_alpha_99))
    print('Use {:.6f} as the maximum allowed value for p when interpreting the 99.9% p-value (alpha = .001).'.format(bonferoni_alpha_99_9))

    status.update(label='✅ Data loaded and group stats ready.', state='complete', expanded=False)

    # Display group stats in Streamlit browser
    st.subheader('Group Statistics')
    st.dataframe(group_stats_df)
    st.write(f'Number of Variables: {number_of_variables} | Number of Tests: {number_of_tests}')
    st.write(f'Bonferroni-adjusted alpha — 95%: {bonferoni_alpha_95:.4f} | 99%: {bonferoni_alpha_99:.5f} | 99.9%: {bonferoni_alpha_99_9:.6f}')

    ###############################################################################################################
    # Step 6: Get the test stats

    st.subheader('Bootstrap Test Results')

    test_df = pd.DataFrame(columns=['Test', 'Pair', 'Group_A', 'Group_B', 'A_N', 'B_N', 'A_CR', 'B_CR', 'A_STD', 'B_STD', 'Diff_B-A', 't', 'p', 't-Significance', 'Boostrap_Mean', 'Lower_95%_CI', 'Upper_95%_CI', 'Lower_99%_CI', 'Upper_99%_CI', 'Lower_99.9%_CI', 'Upper_9.9%_CI', 'Bootstrap_p', 'Bootstrap_Significance'])

    plot_paths = []  # Track saved plot file paths
    total_tests = number_of_tests
    overall_progress = st.progress(0, text='Overall progress: 0 of {} tests'.format(total_tests))
    test_counter = 1

    for i in range(number_of_tests):
        for j in range(i, number_of_variables - 1):

            Group_A  = group_stats_df['Group_ID'][i]
            A_N      = group_stats_df['N'][i]
            A_CR     = group_stats_df['Conversion_Rate'][i]
            A_STD    = group_stats_df['STD'][i]
            Group_B  = group_stats_df['Group_ID'][j + 1]
            B_N      = group_stats_df['N'][j + 1]
            B_CR     = group_stats_df['Conversion_Rate'][j + 1]
            B_STD    = group_stats_df['STD'][j + 1]
            Diff_B_A = B_CR - A_CR
            Pair     = Group_A + '/' + Group_B

            print(f'\n--- Test {test_counter}: Pair {Pair} ---')

            with st.status(f'Test {test_counter}/{total_tests}: {Pair}', expanded=True) as test_status:

                # t-test
                st.write('Running t-test...')
                print('  Running t-test...')
                t, p = ttest(df, Group_A, Group_B)
                print(f'  t-test complete: t={t:.4f}, p={p:.6f}')
                st.write(f'✅ t-test complete — t={t:.4f}, p={p:.6f}')

                t_significance = significance_level(p, bonferoni_alpha_95, bonferoni_alpha_99, bonferoni_alpha_99_9)

                # Bootstrap with inline browser progress bar
                n_int = int(n)
                st.write(f'Running {n_int:,} bootstrap iterations...')
                print('  Running bootstrap sample (this may take a while for large datasets)...')
                bootstrap_progress = st.progress(0, text='Bootstrap: 0%')

                A_df = df.query("group == '" + Group_A + "'")
                B_df = df.query("group == '" + Group_B + "'")

                large_dataset_threshold  = 50000
                large_dataset_sample_size = 10000
                A_sample_size = min(len(A_df), large_dataset_sample_size) if len(A_df) > large_dataset_threshold else len(A_df)
                B_sample_size = min(len(B_df), large_dataset_sample_size) if len(B_df) > large_dataset_threshold else len(B_df)

                if len(A_df) > large_dataset_threshold:
                    st.write(f'ℹ️ Group {Group_A} is large ({len(A_df):,} rows) — capping resample to {large_dataset_sample_size:,} rows per iteration.')
                    print(f'  Group {Group_A} is large ({len(A_df):,} rows). Capping each bootstrap resample to {large_dataset_sample_size:,} rows.')
                if len(B_df) > large_dataset_threshold:
                    st.write(f'ℹ️ Group {Group_B} is large ({len(B_df):,} rows) — capping resample to {large_dataset_sample_size:,} rows per iteration.')
                    print(f'  Group {Group_B} is large ({len(B_df):,} rows). Capping each bootstrap resample to {large_dataset_sample_size:,} rows.')

                differences = []
                for k in range(n_int):
                    if k % (n_int // 10) == 0:
                        pct = int(k / n_int * 100)
                        bootstrap_progress.progress(k / n_int, text=f'Bootstrap: {pct}% ({k:,}/{n_int:,})')
                        print(f'  Bootstrap progress: {k}/{n_int} ({pct}%)')
                    A_sample_cr = A_df.sample(A_sample_size, replace=True)['converted'].mean()
                    B_sample_cr = B_df.sample(B_sample_size, replace=True)['converted'].mean()
                    differences.append(B_sample_cr - A_sample_cr)

                bootstrap_progress.progress(1.0, text='Bootstrap: 100% complete')
                print(f'  Bootstrap complete: {n_int}/{n_int} (100%)')
                differences   = np.array(differences)
                Bootstrap_Mean = differences.mean()
                print(f'  Bootstrap mean: {Bootstrap_Mean:.6f}')
                st.write(f'✅ Bootstrap complete — Mean difference: {Bootstrap_Mean:.6f}')

                # Null hypothesis distribution
                st.write('Creating null hypothesis distribution...')
                null_hypothesis = np.random.normal(0, differences.std(), differences.size)
                Null_Mean = null_hypothesis.mean()
                print(f'  Null mean: {Null_Mean:.6f}')

                # Confidence intervals
                lower_95,   upper_95   = confidence_interval(differences, bonferoni_alpha_95)
                lower_99,   upper_99   = confidence_interval(differences, bonferoni_alpha_99)
                lower_99_9, upper_99_9 = confidence_interval(differences, bonferoni_alpha_99_9)

                bootstrap_p            = bootstrap_p_value(null_hypothesis, Bootstrap_Mean)
                bootstrap_significance = significance_level(bootstrap_p, bonferoni_alpha_95, bonferoni_alpha_99, bonferoni_alpha_99_9)

                test_status.update(label=f'✅ Test {test_counter}/{total_tests}: {Pair} complete', state='complete', expanded=False)

            # Plot is rendered outside the st.status block so it stays visible after the block collapses
            st.subheader(f'Plot: {Pair}')
            print('  Creating plot...')
            plot_path = create_the_plot(differences, null_hypothesis, Bootstrap_Mean, Null_Mean, experiment_name, Pair, plot_output_folder)
            plot_paths.append(plot_path)
            print(f'  Plot saved: {plot_path}')

            # Results summary block under each plot
            st.markdown(
                f"""
| Test | Statistic | p-value | Significance |
|---|---|---|---|
| Regular t-test | t = {t:.5f} | {p:.6f} | {t_significance if t_significance else '(not significant)'} |
| Bootstrap test | Mean diff = {Bootstrap_Mean:.5f} | {bootstrap_p:.6f} | {bootstrap_significance if bootstrap_significance else '(not significant)'} |
"""
            )

            # Add row to results dataframe
            column_data = [test_counter, Pair, Group_A, Group_B, A_N, B_N, A_CR, B_CR, A_STD, B_STD, Diff_B_A,
                           t, p, t_significance, Bootstrap_Mean, lower_95, upper_95, lower_99, upper_99,
                           lower_99_9, upper_99_9, bootstrap_p, bootstrap_significance]
            test_df.loc[len(test_df.index)] = column_data

            overall_progress.progress(test_counter / total_tests,
                                       text=f'Overall progress: {test_counter} of {total_tests} tests')
            test_counter += 1

    print(test_df)

    # Display final results table in browser
    st.subheader('Test Results')
    st.dataframe(test_df)

    # Save results CSV to output folder
    os.makedirs(plot_output_folder, exist_ok=True)
    experiment_name_text = re.sub(r'[\/*?:"<>|]', "_", experiment_name).replace(" ", "_")
    csv_path = plot_output_folder + experiment_name_text + '_results.csv'
    test_df.to_csv(csv_path, index=False)
    print(f'Results saved to: {csv_path}')

    ###############################################################################################################
    # Show the time to run
    end_time = pd.Timestamp.now()
    minutes_elapsed = (end_time - start_time)
    print('Finished parsing in ' + str(minutes_elapsed))
    st.success(f'✅ Analysis complete — finished in {minutes_elapsed}')

    # Print summary of all saved files
    st.subheader('📁 Saved Output Files')
    st.write(f'**Results CSV:** {csv_path}')
    for path in plot_paths:
        st.write(f'**Plot:** {path}')

###############################################################################################################
# The test stats are displayed above and can be copied from the browser table
###############################################################################################################
