#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import os
from gerrychain import Graph, GeographicPartition, Partition, Election, accept
from gerrychain.updaters import Tally, cut_edges
import geopandas as gpd
import numpy as np
from gerrychain.random import random
import copy
import seaborn as sns

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.accept import always_accept
from gerrychain.metrics import polsby_popper
from gerrychain import constraints
from gerrychain.constraints import no_vanishing_districts

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx
import pandas
import math
from itertools import combinations_with_replacement
#from IPython.display import clear_output
from functools import partial


# In[2]:


# setup -- SLOW
#PA
shapefile = "https://github.com/mggg-states/PA-shapefiles/raw/master/PA/PA_VTD.zip"

#NC
#shapefile = "https://github.com/mggg-states/NC-shapefiles/raw/master/NC_VTD.zip"

df = gpd.read_file(shapefile)


#For Pennsylvania
county_col = "COUNTYFP10"
pop_col = "TOT_POP"
uid = "GEOID10"

#FOR North Carolina
#county_col = "County"
#uuid = "VTD"

graph = Graph.from_geodataframe(df,ignore_errors=True)
graph.add_data(df,list(df))
graph = nx.relabel_nodes(graph, df[uid])
counties = (set(list(df[county_col])))
countydict = dict(graph.nodes(data=county_col))


#print(counties)
#print(countydict)


# In[3]:


totpop = 0
num_districts = 18
for n in graph.nodes():
    graph.node[n]["TOT_POP"] = int(graph.node[n]["TOT_POP"])
    totpop += graph.node[n]["TOT_POP"]


# In[4]:


starting_partition = GeographicPartition(
    graph,
    assignment="GOV",
    updaters={
        "polsby_popper" : polsby_popper,
        "cut_edges": cut_edges,
        "population": Tally(pop_col, alias="population"),

    }
)


# In[5]:


def county_splits_dict(partition):
    ''' 
    Args:
        partition: a graph partition
    
    Returns:
        county splits: a dictionary with keys as district numbers and values Counter() dictionaries
                        these counter dictionaries have pairs COUNTY_ID : NUM which counts the number of VTDS
                        in the county in the district
        
    '''
    
    county_splits = {k:[] for k in counties}
    county_splits = {   k:[countydict[v] for v in d] for k,d in partition.assignment.parts.items()   }
    county_splits = {k: Counter(v) for k,v in county_splits.items()}
    return county_splits


# In[6]:


def district_splits_dict(county_splits):
    '''
    Args:
        county_splits: a dictionary with keys as district numbers and values Counter() dictionaries
                        these counter dictionaries have pairs COUNTY_ID : NUM which counts the number of VTDS
                        in the county in the district 
        
    Returns:
       district_splits: a dictionary that has as keys the county id and returns as values the
    districts in that county. 
    

    '''
    district_splits = {k:[] for k in counties}
    
    for county in counties:
        districts = {}
        for district in county_splits.keys():
            if county in county_splits[district].keys():
                district_splits[county].append(district)
    return district_splits            
            


def reverse_countydict():
    '''
    Returns a dictionary that maps a county to a list of VTDs in that county 
    '''
    rev = {k:[] for k in counties}
    for county in counties:
        for vtd in countydict.keys():
            if countydict[vtd] == county:
                rev[county].append(vtd)
    return rev
            
# In[7]:


# various functions to measure splits according to the proposed PA rule. Feel free to ignore

def pieces_allowed():
    district_splits ={}
    
    for county in counties:
        sg=graph.subgraph(n for n, v in graph.nodes(data=True) if v[county_col]==county)
        pop = 0;
        
        for n in sg.nodes():
            pop += sg.node[n]["TOT_POP"]
        
        district_splits[county] = math.ceil(pop/(totpop/num_districts)) + 1
    return district_splits

def other_pieces_allowed():
    district_splits ={}
    
    for county in counties:
        sg=graph.subgraph(n for n, v in graph.nodes(data=True) if v[county_col]==county)
        pop = 0;
        
        for n in sg.nodes():
            pop += sg.node[n]["TOT_POP"]
        
        district_splits[county] = math.ceil(pop/(totpop/num_districts))
    return district_splits

def too_many_pieces(partition):
    district_splits = district_splits_dict(county_splits_dict(partition))
    pieces = pieces_allowed()
    too_many = 0
    
    for county in counties:
        if len(district_splits[county]) > pieces[county]:
            too_many += 1
    
    return too_many

def other_too_many_pieces(partition):
    district_splits = district_splits_dict(county_splits_dict(partition))
    pieces = other_pieces_allowed()
    too_many = 0
    
    for county in counties:
        if len(district_splits[county]) > pieces[county]:
            too_many += 1
    
    return too_many

def how_many_more(partition):
    district_splits = district_splits_dict(county_splits_dict(partition))
    pieces = pieces_allowed()
    too_many = 0
    
    for county in counties:
        if len(district_splits[county]) > pieces[county]:
            too_many += len(district_splits[county]) - pieces[county]
    return too_many


# In[8]:


def cut_in_county(part,sg):
    '''
    Args:
        part: a partition of a graph
        sg: a subgraph of a graph
        
    Returns:
        num_ce_in_count: the number of cut edges in the partition that are in the subgraph. ###################################
    '''
    num_ce_in_count = 0
    for edge in part["cut_edges"]:
         if edge in sg.edges():
            num_ce_in_count += 1
    return num_ce_in_count


# In[9]:


def our_split_score_1(part):
    '''
    Args:
       part: a parition of a graph. 
        
    Returns:
        sum: sum over the counties of the portion of cut edges over the total number of edges in the county. #####################
    '''
    
    sum = 0
    
    for county in counties:
        sg=graph.subgraph(n for n, v in graph.nodes(data=True) if v[county_col]==county)
        sum += cut_in_county(part,sg) / len(sg.edges())
      
    return sum


# In[10]:


def our_split_score_2(part):
     '''
    Args:
       part: Takes in a parition of a graph. 
        
    Returns:
        p : the portion of cut edges whose nodes are in different counties
        over the total number of cut edges.
    '''
    
    ce_btn_counties = 0
    
    for ce in part["cut_edges"]:
        if int(countydict[str(ce[0])]) != int(countydict[str(ce[1])]):
            ce_btn_counties += 1
    
    p = ce_btn_counties / len(part["cut_edges"])

    return p


# In[11]:

def vtds_per_county(county_splits):
    
     ''' 
     A function that creates a dictionary of counties mapped to the value that is the
     the number of vtds in the county.
    Args:
        county_splits: a dictionary with keys as district numbers and values Counter() dictionaries
                        these counter dictionaries have pairs COUNTY_ID : NUM which counts the number of VTDS ##########################
                        in the county in the district   
    Returns:
        vtds: a dictionary with keys as COUNTIES and values as NUM OF VTDS 
    '''
    
    vtds = {}
    
    for counter in county_splits.values():
        for county in counter.keys():
            if county in vtds:
                vtds[county] += counter[county]
            else:
                vtds[county] = counter[county]
    return vtds


# In[12]:


def pops_per_county(county_splits,rev):
     '''
     A function that finds the population fo a county
    Args:
        county_splits: a dictionary with keys as district numbers and values Counter() dictionaries
                        these counter dictionaries have pairs COUNTY_ID : NUM which counts the number of VTDS
                        in the county in the district 
        rev: a dictionary that maps counties to a list of vtds in the county.
            
    Returns:
        pops: an integer value of the population within a county.
    '''
    pops = {}
    
    for county in rev.keys():
        pop = 0
        for vtd in rev[county]:
            pop += graph.nodes[vtd]["TOT_POP"]
        pops[county] = pop
    return pops


# In[13]:
def vtds_per_district(county_splits):
     '''
     A function that counts the number of VTDs per district.
    Args:
       county_splits: a dictionary with keys as district numbers and values Counter() dictionaries.
                        The Counter dictionaries have pairs COUNTY_ID : NUM which counts the number of VTDS
                        in the county in the district
        
    Returns:
       vtds: the total number of vtds in a district.
    '''
    
    vtds = {}
    
    for district in county_splits.keys():
        sum = 0
        counter = county_splits[district]
        for vtd in counter.values():
            sum += vtd
        vtds[district] = sum
    return vtds        


# In[14]:


def pops_per_district(partition):
     '''
     A function that calculates the population of a district in a plan.
     
    Args:
       partition: paritition of a graph 
        
    Returns:
       pops: population of a district/partition
    '''
    dictionary = dict(partition.assignment)
    pops = {}
    
    for i in range(num_districts):
        for vtd in dictionary.keys():
            if i+1 in pops and dictionary[vtd] == i + 1:
                pops[i+1] += graph.nodes[vtd]["TOT_POP"]
            elif dictionary[vtd] == i+1:
                pops[i+1] = graph.nodes[vtd]["TOT_POP"]
    return pops


# In[15]:


def total_vtds(vtds):
     '''
     A function that counts the number of vtds from a dictionary of counties.
    Args:
       vtds: a dictionary of COUNTY: VTDS
        
    Returns:
        total: the total number of VTDS  in a county
    '''
    total = 0
    
    for county in vtds.keys():
        total += vtds[county]
    return total


# In[16]:


def total_pops(pops):
     '''
     A function that counts the total population over counties  
    Args:
       pops: a dictionary of COUNTY: TOTAL POPULATION 
        
    Returns:
        total: total population
    '''
    total = 0
    for pop in pops.values():
        total += pop
    return total


# In[17]:


def p_i_given_j(county_splits, vtds, district_i, county_j):
     '''
    Args:
       county_splits: a dictionary that counts the number of splits in a county.
       vtds: a dictionary of DISTRICT:{ COUNTY: VTDS }
       district_i: a district
       county_j:   a county
        
    Returns:
        p: the proportion of vtds of county j given that you're in district_i.
    '''
    
    counter = county_splits[district_i]
    intersection = counter[str(county_j)]
    p = intersection / vtds[str(county_j)]
    return p 

# In[18]:


#def q_j(vtds_d,county_j,total):
#    return vtds[county_j] / total


# In[19]:


def power_entropy(county_splits,vtds,total,alpha):
     '''
    Args:
       county_splits: a dictionary COUNTY : NUM OF SPLITS
       vtds: a dictionary of COUNTY: VTDS 
       total: a population count
       alpha: a value
        
    Returns:
        entropy: an entropy score value
    '''
    entropy = 0
    for county_j in counties:
        inner_sum = 0
        q = q_j(vtds,county_j,total)
        for district_i in range(num_districts):
            p = p_i_given_j(county_splits, vtds,district_i+1,county_j)
            inner_sum += p ** (1-alpha)
        entropy += 1/q * (inner_sum-1)
        #print(1/q * (inner_sum-1))
    return entropy


# In[20]:


#def Shannon_entropy(county_splits, vtds, total):
#    entropy = 0
#    for county_j in counties:
#        inner_sum = 0
#        q = q_j(vtds,county_j,total)
#        for district_i in range(num_districts):
#            p = p_i_given_j(county_splits, vtds,district_i+1,county_j)
#            if p != 0:
#                inner_sum += p * math.log(1/p)
#        entropy += q * (inner_sum)
#        #print(1/q * (inner_sum-1))
#    return entropy


# In[21]:


#def p_i(vtds,district_i,total):
#    return vtds[district_i] / total


# In[22]:


#def q_j_given_i(county_splits, vtds_d, district_i, county_j):
#    counter = county_splits[district_i]
#    intersection = counter[str(county_j)]
#    
#    return intersection / vtds_d[district_i]


# In[23]:


#def other_power_entropy(county_splits,vtds_d,total,alpha):
#    entropy = 0
#    for district_i in range(num_districts):
#        innersum = 0
#        p = p_i(vtds_d,district_i+1,total)
#        for county_j in counties:
#            q = q_j_given_i(county_splits,vtds_d,district_i+1,county_j)
#            innersum += q ** (1-alpha)
#        entropy += 1/p * (innersum-1)
#    return entropy


# In[24]:


#def symmetric_power_entropy(county_splits,vtds_c,vtds_d,total,alpha):
#    return power_entropy(county_splits,vtds_c,total,alpha) + other_power_entropy(county_splits,vtds_d,total,alpha)


# In[25]:


def edge_entropy(partition):
    ''' 
    A function that calculates the entropy score
    Args: 
        parition: a partition of a graph
    Returns:
        entropy: the edge entropy score of 
        ''''
    entropy = 0
    total_edges = len(graph.edges())
    countynodelist = {
        county: frozenset(
            [node for node in graph.nodes() if graph.nodes[node][county_col] == county]) for county in counties
    }
    districts_in_counties = {
        county: frozenset([partition.assignment[d] for d in countynodelist[county]]) for county in counties
    }
    for county in counties:
        county_subgraph = graph.subgraph([n for n in graph.nodes if graph.nodes[n][county_col] == county])
        county_edges = len(county_subgraph.edges())
        for (district1, district2) in combinations_with_replacement(districts_in_counties[county],2):
            p_ij = len([e for e in county_subgraph.edges() if set(
                [partition.assignment[e[0]], partition.assignment[e[1]]]) == set([district1, district2])])
            p_ij = p_ij/len(county_subgraph.edges())
            if (p_ij != 0):
                entropy -= p_ij*np.log(p_ij)*county_edges/total_edges
    return entropy

def num_of_splittings(partition):
    dictionary = county_splits_dict(partition)
    counter = 0
    for district in dictionary.keys():
        counter += len(dictionary[district])
    return counter


# In[38]:


# In[27]:


# calculates the population of a given county

def county_pop(rev, county_j):
    ''' 
    A function that returns 
    Args: 
        rev: a dictionary mapping counties to VTDs in the counties 
        county_j: a county from the shapefile
    Returns:
        pop: population of the passed county
        ''''
    pop = 0
    for vtd in rev[county_j]:
        pop += graph.nodes[vtd]["TOT_POP"]
    return pop


# In[28]:


# calculates population of given district

def district_pop(part, district_i):
    ''' 
    Args:
        part: a graph parition 
        district_i: a district in from the shapefile. 
        
    Returns:
        pop: 
    ''''
    pop = 0
    for vtd in dict(part.parts)[district_i]:
        pop += graph.nodes[vtd]["TOT_POP"]
    return pop


# In[29]:


#calculates population of intersection of given district and county

def intersection_pop(part, county_vtds, county_j, district_i):
    
    
    intersection = [vtd for vtd in county_vtds[county_j] if vtd in dict(part.parts)[district_i]]
    
    pop = 0
    for vtd in intersection:
        pop += graph.nodes[vtd]["TOT_POP"]
    return pop


# In[30]:


#calculates power entropy

def power_entropy(partition, rev, alpha):
    ''' 
    Args:
        partition: a graph parition
        rev: a dictionary mapping counties to VTDs in the county ####################################
        alpha: a value 
        
    Returns:
        entropy: Returns Power Entropy score
    ''''
    entropy = 0
    for county_j in counties:
        inner_sum = 0
        cpop = county_pop(rev,county_j)
        q = cpop / totpop
        for district_i in range(num_districts):
            p = intersection_pop(partition,rev,county_j,district_i+1) / cpop
            inner_sum += p ** (1 - alpha)
        entropy += 1 / q * (inner_sum - 1)
    return entropy


# In[31]:


# calculates Shannon entropy

def Shannon_entropy(partition, rev): 
    ''' 
    Args:
        partition: a graph parition
        rev: a dictionary mapping counties to VTDs in the county ####################################
        
    Returns:
        entropy: Returns Shannon Entropy score
    ''''
    entropy = 0
    for county_j in counties:
        inner_sum = 0
        cpop = county_pop(rev,county_j)
        q = cpop / totpop
        for district_i in range(num_districts):
            p = intersection_pop(partition,rev,county_j,district_i+1) / cpop 
            if p != 0:
                inner_sum += p * math.log(1/p)
        entropy += q * inner_sum
    return entropy


# In[32]:


#county_vtds = county_splits_dict(starting_partition)
#print(power_entropy(county_splits,vtds,total,0.5))
rev = reverse_countydict() 

#print(Shannon_entropy(starting_partition,rev))


# In[33]:


d = county_splits_dict(starting_partition)
sum( [ len([ dd for dd  in [dict(v) for v in d.values()] if k in dd.keys()]) > 1 for k in counties] )
#print(our_split_score_1(starting_partition))
#print(our_split_score_2(starting_partition))
#print(too_many_pieces(starting_partition))
#print(other_too_many_pieces(starting_partition))
#print(district_splits_dict(county_splits))


# In[34]:


proposal = partial(
        recom, pop_col="TOT_POP", pop_target=totpop/num_districts, epsilon=0.02, node_repeats=1
    )

compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]), 2 * len(starting_partition["cut_edges"])
    )

chain = MarkovChain(
        proposal,
        constraints=[
            constraints.within_percent_of_ideal_population(starting_partition, 0.05),compactness_bound
          #constraints.single_flip_contiguous#no_more_discontiguous
        ],
        accept=accept.always_accept,
        initial_state=starting_partition,
        total_steps=5000
    )


# In[37]:


score_1 = []
score_2 = []
pieces = []
power = []
Shannon = []
cuts = []
splittings = []

t = 0
i = 1
for part in chain:
    if i > 100:    
        county_vtds = county_splits_dict(part)
        #score_1.append(our_split_score_1(part))
        #score_2.append(our_split_score_2(part))
        #pieces.append(too_many_pieces(part))
        #county_splits = county_splits_dict(part)
        #vtds = vtds_per_county(county_splits)
        #total = total_vtds(vtds)
        cuts.append(len(part["cut_edges"]))
        splittings.append(num_of_splittings(part))
        power.append(power_entropy(part,rev,4/5))
        Shannon.append(Shannon_entropy(part,rev))
    
        t += 1
        if t % 100 == 0:
            print("finished chain " + str(t))
        else:
            i+= 1
            t+=1
            
np.savetxt("PA_cuts.txt", cuts)
np.savetxt("PA_splittings.txt", splittings)
np.savetxt("PA_power_entropy.txt", power)
np.savetxt("PA_Shannon_entropy.txt",Shannon)


# In[36]:


#colors = ['hotpink']
#labels = ['VTD']
#plt.figure()
#for i in range(1):
#    sns.distplot(score_1,kde=False, color=colors[i],label=labels[i])
#plt.legend()
#plt.xlabel("Score 1")
#plt.show()

#plt.figure()
#for i in range(1):
#    sns.distplot(score_2,kde=False, color=colors[i],label=labels[i])
#plt.legend()
#plt.xlabel("Score 2")
#plt.show()

#plt.figure()
#for i in range(1):
#    sns.distplot(pieces,kde=False, color=colors[i],label=labels[i])
#plt.legend()
#plt.xlabel("Too Many Splits")
#plt.show()

#plt.figure()
#for i in range(1):
#    sns.distplot(power,kde=False, color=colors[i],label=labels[i])
#plt.legend()
#plt.xlabel("Power Entropy (alpha = 4/5)")
#plt.show()

#plt.figure()
#for i in range(1):
#    sns.distplot(Shannon,kde=False, color=colors[i],label=labels[i])
#plt.legend()
#plt.xlabel("Shannon Entropy")
#plt.show()


# In[ ]:


#CHANGE
plt.figure()
plt.hist(power)
plt.title("Histogram of Power Entropy (Alpha = 0.8)")
plt.xlabel("Power Entropy")
plt.savefig("PA_hist_power_entropy_5000.png")
plt.show()

plt.figure()
plt.hist(Shannon)
plt.title("Histogram of Shannon Entropy")
plt.xlabel("Shannon Entropy")
plt.savefig("PA_hist_Shannon_entropy_5000.png")
plt.show()

#CHANGE
plt.figure()
plt.scatter(splittings, power)
plt.title("Power Entropy (Alpha = 0.8) vs. Number of Splits")
plt.xlabel('Splits')
plt.ylabel('Power Entropy')
plt.xlim(min(splittings) - 5, max(splittings) + 5)
plt.savefig("PA_scatter_power_entropy_splits_5000.png")
plt.show()

plt.figure()
plt.scatter(splittings, Shannon)
plt.title("Shannon Entropy vs. Number of Splits")
plt.xlabel('Splits')
plt.ylabel('Shannon Entropy')
plt.xlim(min(splittings) - 5, max(splittings) + 5)
plt.savefig("PA_scatter_Shannon_entropy_splits_5000.png")
plt.show()

#CHANGE
plt.figure()
plt.scatter(cuts, power)
plt.title("Power Entropy (Alpha = 0.8) vs. Number of Cut Edges")
plt.xlabel('Cut Edges')
plt.ylabel('Power Entropy')
plt.savefig("PA_scatter_power_entropy_cut_edges_5000.png")
plt.show()

plt.figure()
plt.scatter(cuts, Shannon)
plt.title("Shannon Entropy vs. Number of Cut Edges")
plt.xlabel('Cut Edges')
plt.ylabel('Shannon Entropy')
plt.savefig("PA_scatter_Shannon_entropy_cut_edges_5000.png")
plt.show()


# In[ ]:




