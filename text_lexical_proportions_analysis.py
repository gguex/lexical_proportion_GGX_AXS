# -*- coding: utf-8 -*-
"""
Code for the article "Vers une estimation robuste des proportions lexicales"
(submitted to JADT 2020)

This code will analyse a text file (by default, the Jule Vernes novel
"De la Terre à la Lune" found on http://www.gutenberg.org/files/799/799-0.txt)
to compute lexical proportions for different word properties. It will need a file
containing french stopwords (we use here the file provided by Jacques Savoy
http://members.unine.ch/jacques.savoy/clef/frenchST.txt).

The 3 lexical proportions studied in this code are :

    1. The proportion of stopwords
    2. The proportion of words longer than the mean length
    3. The proportion of words with an even length

@author: Guillaume Guex & Aris Xanthos
@date: january 2020
"""

##########################################################
########################### HEADER #######################
##########################################################

# Libraries
import random
import nltk
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
from progress.bar import Bar

# Parameters

# The name of the text file
input_filename = "JulesVerneGutenberg.txt"
# The name of the file containing stopwords
stopword_filename = "stopwords-fr-savoy.txt"
# The number of subsamples for each text lengths
n_sample = 100
# The step increases (and initial value) for text lengths
step_graphs = 5000
# The path of the local folder (containing files)
local_folder = "./"
# The default font size for the output figures
font_size = 15
# The sampling rate of the initial text (to reduce computing time by taking
# a smaller initial sample)
sampling_rate = 1
# Option to compute O(2) approximations (warning: computing complexity will
# become quadratic instead of linear)
compute_o2 = False

##########################################################s
########### LOADING TEXT AND PREPROCESSING ###############
##########################################################

# Opening and processing stopwords
with open(local_folder + stopword_filename, "r", encoding="utf-8") as stop_word_file:
    stop_word_list = stop_word_file.read()
    stop_word_list = stop_word_list.split("\n")

# Opening the file and store data in "raw"
with open(local_folder + input_filename, encoding="utf-8") as text_file:
    raw = text_file.read()

# Removing punctuation
raw = raw.translate(str.maketrans('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                                          " "*len('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')))
# Transforming the text in token list
token_list = nltk.word_tokenize(raw.strip())

# Sampling
assert 0 < sampling_rate > 0 <= 1
if sampling_rate < 1:
    token_list = random.sample(token_list, int(sampling_rate*len(token_list)))

# Removing non-alpha token and put it in lower cases
token_list = [token.lower() for token in token_list if token.isalpha()]
n = len(token_list)

# Creating a numerical vector of token_list
ntoken_list = pd.factorize(token_list)[0]

# Creating a dictonnary with vocabulary and word frequencies
lexique_dict = nltk.FreqDist(token_list)
w =  len(lexique_dict)
lexique_dict_values_list = list(lexique_dict.values())

print("#types: %i, #tokens: %i" % (w, n))

##########################################################
########### DEFINITION OF PROPERTIES #####################
##########################################################

# Is a stopword
property_1_list = [word for word in lexique_dict.keys() if word in stop_word_list]
has_property_1_list = [(word in stop_word_list) for word in lexique_dict.keys()]

# longer than average
average_len = sum(len(word) for word in lexique_dict.keys())/w
print("Average length:", average_len)
property_2_list = [word for word in lexique_dict.keys() if len(word) > average_len]
has_property_2_list = [int(len(word) > average_len) for word in lexique_dict.keys()]

# Even length
property_3_list = [word for word in lexique_dict.keys() if len(word) % 2 == 0]
has_property_3_list = [int(len(word) % 2 == 0) for word in lexique_dict.keys()]

# Number of words with the property in text
w_p_1 = sum(has_property_1_list)
w_p_2 = sum(has_property_2_list)
w_p_3 = sum(has_property_3_list)

print("#p1: %i, #p2: %i, #p3: %i" % (w_p_1, w_p_2, w_p_3))

##########################################################
##################### SUBSAMPLING  #######################
##########################################################

# The list of tested sample sizes
sub_size_list = list(range(step_graphs, n, step_graphs))
sub_size_list.append(n)

theoretical_ratio_1_list = []
theoretical_ratio_2_list = []
theoretical_ratio_3_list = []
real_ratio_1_list = []
real_ratio_2_list = []
real_ratio_3_list = []
mean_ratio_1_list = []
mean_ratio_2_list = []
mean_ratio_3_list = []
std_ratio_1_list = []
std_ratio_2_list = []
std_ratio_3_list = []
if compute_o2:
    theoretical_ratio2_1_list = []
    theoretical_ratio2_2_list = []
    theoretical_ratio2_3_list = []
for m in sub_size_list:
    print("%i/%i" % (m/step_graphs, len(sub_size_list)))

    #### Theoretical computation

    pi = []
    pij = np.zeros(shape=(len(lexique_dict),len(lexique_dict)))
    with Bar("Calcul des pi_ij", max=len(lexique_dict)) as bar:
        for i in range(len(lexique_dict)):
            # Computing n_i
            n_i = lexique_dict_values_list[i]
            # Computing p_i
            pi.append( hypergeom.pmf(0, n, n_i, m) )
            #  If O(2), computing p_ij
            if compute_o2:
                for j in range(i+1, len(lexique_dict)):
                    n_j = lexique_dict_values_list[j]
                    pij[i,j] = hypergeom.pmf(0, n, n_i + n_j, m)
            bar.next()

        # If O(2), filling p_ij
        if compute_o2:
            pij = pij + np.transpose(pij)
            np.fill_diagonal(pij, pi)

    # Theoretical variety
    V = w - sum(pi)
    # Theoretical variety of properities
    V_p_1 = w_p_1 - sum(np.array(pi) * np.array(has_property_1_list))
    V_p_2 = w_p_2 - sum(np.array(pi) * np.array(has_property_2_list))
    V_p_3 = w_p_3 - sum(np.array(pi) * np.array(has_property_3_list))
    # Storing ratios
    theoretical_ratio_1_list.append(V_p_1 / V)
    theoretical_ratio_2_list.append(V_p_2 / V)
    theoretical_ratio_3_list.append(V_p_3 / V)

    # If O(2)
    if compute_o2:
        # Theoretical variance of V
        P_mat = pij - np.outer(pi,pi)
        var_v = np.sum(P_mat)
        # Theoretical covariance between V and V_p
        cov_vvp_1 =  np.sum(P_mat[:,has_property_1_list])
        cov_vvp_2 =  np.sum(P_mat[:,has_property_2_list])
        cov_vvp_3 =  np.sum(P_mat[:,has_property_3_list])
        # O(2) estimation of ratio
        theoretical_ratio2_1_list.append(V_p_1 / V - cov_vvp_1 / V**2 + var_v*V_p_1 / V**3)
        theoretical_ratio2_2_list.append(V_p_2 / V - cov_vvp_2 / V**2 + var_v*V_p_2 / V**3)
        theoretical_ratio2_3_list.append(V_p_3 / V - cov_vvp_3 / V**2 + var_v*V_p_3 / V**3)

    #### Subsampling
    sample_ratio_1_list = []
    sample_ratio_2_list = []
    sample_ratio_3_list = []
    for _ in range(n_sample):
        # Drawing a sample without replacement
        num_sample_index = np.random.choice(ntoken_list, m, replace=False)
        # Get the unique voc numbers
        unique_num_sample_index = np.unique(num_sample_index)
        # Computing variety of the sample
        V_sample = len(unique_num_sample_index)
        # Counting the words with properties
        V_p_sample_1 = sum(np.array(has_property_1_list)[unique_num_sample_index])
        V_p_sample_2 = sum(np.array(has_property_2_list)[unique_num_sample_index])
        V_p_sample_3 = sum(np.array(has_property_3_list)[unique_num_sample_index])
        # Storing the ratios for the sample
        sample_ratio_1_list.append(V_p_sample_1 / V_sample)
        sample_ratio_2_list.append(V_p_sample_2 / V_sample)
        sample_ratio_3_list.append(V_p_sample_3 / V_sample)
    # Storing the ratios for sample size
    real_ratio_1_list.extend(sample_ratio_1_list)
    real_ratio_2_list.extend(sample_ratio_2_list)
    real_ratio_3_list.extend(sample_ratio_3_list)
    mean_ratio_1_list.append(np.mean(sample_ratio_1_list))
    mean_ratio_2_list.append(np.mean(sample_ratio_2_list))
    mean_ratio_3_list.append(np.mean(sample_ratio_3_list))
    std_ratio_1_list.append(np.std(sample_ratio_1_list))
    std_ratio_2_list.append(np.std(sample_ratio_2_list))
    std_ratio_3_list.append(np.std(sample_ratio_3_list))

##########################################################
####################### PLOTTING #########################
##########################################################


from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rc('font', size=font_size)          # controls default text sizes

### Plotting w/o estimation

# Duplicating the sample size list by the number of time sampling is done (x axis plot)
dup_sub_size_list = np.repeat(sub_size_list, n_sample)

plt.figure("1a")
y1 = np.array(mean_ratio_1_list) + 2*np.array(std_ratio_1_list)/np.sqrt(n_sample-1)
y2 = np.array(mean_ratio_1_list) - 2*np.array(std_ratio_1_list)/np.sqrt(n_sample-1)
plt.fill_between(sub_size_list, y1, y2, facecolor='silver')
plt.scatter(dup_sub_size_list, real_ratio_1_list, s=0.1, facecolor='dimgray')
plt.xlabel("Nombre de tokens")
plt.ylabel("Prop. lex. de stopwords")
plt.show()
plt.figure("1b")
y1 = np.array(mean_ratio_2_list) + 2*np.array(std_ratio_2_list)/np.sqrt(n_sample-1)
y2 = np.array(mean_ratio_2_list) - 2*np.array(std_ratio_2_list)/np.sqrt(n_sample-1)
plt.fill_between(sub_size_list, y1, y2, facecolor='silver')
plt.scatter(dup_sub_size_list, real_ratio_2_list, s=0.1, facecolor='dimgray')
plt.xlabel("Nombre de tokens")
plt.ylabel("Prop. lex. de mots longs")
plt.show()
plt.figure("1c")
y1 = np.array(mean_ratio_3_list) + 2*np.array(std_ratio_3_list)/np.sqrt(n_sample-1)
y2 = np.array(mean_ratio_3_list) - 2*np.array(std_ratio_3_list)/np.sqrt(n_sample-1)
plt.fill_between(sub_size_list, y1, y2, facecolor='silver')
plt.scatter(dup_sub_size_list, real_ratio_3_list, s=0.1, facecolor='dimgray')
plt.xlabel("Nombre de tokens")
plt.ylabel("Prop. lex. de mots de longueur paire")
plt.show()

### Plotting results

# Duplicating the sample size list by the number of time sampling is done (x axis plot)
dup_sub_size_list = np.repeat(sub_size_list, n_sample)

plt.figure("3a")
y1 = np.array(mean_ratio_1_list) + 2*np.array(std_ratio_1_list)/np.sqrt(n_sample-1)
y2 = np.array(mean_ratio_1_list) - 2*np.array(std_ratio_1_list)/np.sqrt(n_sample-1)
plt.fill_between(sub_size_list, y1, y2, facecolor='silver')
plt.scatter(dup_sub_size_list, real_ratio_1_list, s=0.1, facecolor='dimgray')
plt.plot(sub_size_list, theoretical_ratio_1_list, color="black")
if compute_o2:
    plt.plot(sub_size_list, theoretical_ratio2_1_list, "--", color="black")
plt.xlabel("Nombre de tokens")
plt.ylabel("Prop. lex. de stopwords")
plt.show()
plt.figure("3b")
y1 = np.array(mean_ratio_2_list) + 2*np.array(std_ratio_2_list)/np.sqrt(n_sample-1)
y2 = np.array(mean_ratio_2_list) - 2*np.array(std_ratio_2_list)/np.sqrt(n_sample-1)
plt.fill_between(sub_size_list, y1, y2, facecolor='silver')
plt.scatter(dup_sub_size_list, real_ratio_2_list, s=0.1, facecolor='dimgray')
plt.plot(sub_size_list, theoretical_ratio_2_list, color="black")
if compute_o2:
    plt.plot(sub_size_list, theoretical_ratio2_2_list, "--", color="black")
plt.xlabel("Nombre de tokens")
plt.ylabel("Prop. lex. de mots longs")
plt.show()
plt.figure("3c")
y1 = np.array(mean_ratio_3_list) + 2*np.array(std_ratio_3_list)/np.sqrt(n_sample-1)
y2 = np.array(mean_ratio_3_list) - 2*np.array(std_ratio_3_list)/np.sqrt(n_sample-1)
plt.fill_between(sub_size_list, y1, y2, facecolor='silver')
plt.scatter(dup_sub_size_list, real_ratio_3_list, s=0.1, facecolor='dimgray')
plt.plot(sub_size_list, theoretical_ratio_3_list, color="black")
if compute_o2:
    plt.plot(sub_size_list, theoretical_ratio2_3_list, "--", color="black")
plt.xlabel("Nombre de tokens")
plt.ylabel("Prop. lex. de mots de longueur paire")
plt.show()

### Computing MAE and MSE of the two theoretical values vs empirical mean (if computed)

if compute_o2:
    mae_1 = np.mean(np.abs(np.array(mean_ratio_1_list) - np.array(theoretical_ratio_1_list)))
    mse_1 = np.mean((np.array(mean_ratio_1_list) - np.array(theoretical_ratio_1_list))**2)
    mae2_1 = np.mean(np.abs(np.array(mean_ratio_1_list) - np.array(theoretical_ratio2_1_list)))
    mse2_1 = np.mean((np.array(mean_ratio_1_list) - np.array(theoretical_ratio2_1_list))**2)
    mae_2 = np.mean(np.abs(np.array(mean_ratio_2_list) - np.array(theoretical_ratio_2_list)))
    mse_2 = np.mean((np.array(mean_ratio_2_list) - np.array(theoretical_ratio_2_list))**2)
    mae2_2 = np.mean(np.abs(np.array(mean_ratio_2_list) - np.array(theoretical_ratio2_2_list)))
    mse2_2 = np.mean((np.array(mean_ratio_2_list) - np.array(theoretical_ratio2_2_list))**2)
    mae_3 = np.mean(np.abs(np.array(mean_ratio_3_list) - np.array(theoretical_ratio_3_list)))
    mse_3 = np.mean((np.array(mean_ratio_3_list) - np.array(theoretical_ratio_3_list))**2)
    mae2_3 = np.mean(np.abs(np.array(mean_ratio_3_list) - np.array(theoretical_ratio2_3_list)))
    mse2_3 = np.mean((np.array(mean_ratio_3_list) - np.array(theoretical_ratio2_3_list))**2)

    print("Stopwords| O(1) - mae: %E, mse: %E | O(2) - mae: %E, mse: %E" % (mae_1, mse_1, mae2_1, mse2_1))
    print("Mots longs| O(1) - mae: %E, mse: %E | O(2) - mae: %E, mse: %E" % (mae_2, mse_2, mae2_2, mse2_2))
    print("Longueur paire| O(1) - mae: %E, mse: %E | O(2) - mae: %E, mse: %E" % (mae_3, mse_3, mae2_3, mse2_3))

### Plotting properties

lexique_o_dict = {k: v for k, v in sorted(lexique_dict.items(), key=lambda item: item[1], reverse=True)}

has_property_1_index = [index for index, word in enumerate(lexique_o_dict.keys()) if word in stop_word_list]
has_property_2_index = [index for index, word in enumerate(lexique_o_dict.keys()) if len(word) > average_len]
has_property_3_index = [index for index, word in enumerate(lexique_o_dict.keys()) if len(word) % 2 == 0]

n_bins = 20

plt.figure("2a")
plt.hist(has_property_1_index, bins=n_bins, color="silver")
plt.xlabel("Rang")
plt.ylabel("Nombre de stopwords")
plt.show()
plt.figure("2b")
plt.hist(has_property_2_index, bins=n_bins, color="silver")
plt.xlabel("Rang")
plt.ylabel("Nombre de mots longs")
plt.show()
plt.figure("2c")
plt.hist(has_property_3_index, bins=n_bins, color="silver")
plt.xlabel("Rang")
plt.ylabel("Nombre de mots de longueur paire")
plt.show()
