#import
import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election, metrics)
from gerrychain.proposals import recom
from gerrychain.updaters import cut_edges
from functools import partial
import pandas as pd
import networkx as nx
import numpy as np
import geopandas as gpd
import shapely
import scipy
from scipy.stats.stats import pearsonr
import os
import csv
from itertools import combinations

from score_functions import(
    locality_splits_dict,
    num_parts,
    coincident_boundaries,
    vtds_per_district,
    shannon_entropy,
    power_entropy,
    invert_dict,
    num_split_localities,
    pieces_allowed,
    pennsylvania_fouls, 
    vtds_to_localities,
    dictionary_to_score,
    symmetric_entropy,
    num_pieces
)

outdir = "./pa_score_outputs/"
try:
    # Create target Directory
    os.mkdir(outdir)
    print("Directory " , outdir ,  " created ") 
except FileExistsError:
    print("Directory " , outdir ,  " already exists")
print()

#load the dual graph
graph = Graph.from_file("./notebooks/PA_VTD/PA_VTD.shp")
gdf = gpd.read_file("./notebooks/PA_VTD/PA_VTD.shp")

#elections and updaters
pop_col = "TOT_POP"
election_names = [
    "PRES12",
    "PRES16",
    "SENW101216",
]
election_columns = [
    ["PRES12D", "PRES12R"],
    ["T16PRESD", "T16PRESR"],
    ["W101216D", "W101216R"],
]
updaters1 = {
    "population": updaters.Tally("TOT_POP", alias="population"),
    "cut_edges": cut_edges,
}

elections = [
    Election(
        election_names[i],
        {"Democratic": election_columns[i][0], "Republican": election_columns[i][1]},
    )
    for i in range(len(election_names))
]
election_updaters = {election.name: election for election in elections}
updaters1.update(election_updaters)

#fix some strings which should be numbers
for n in graph.nodes():
    graph.nodes[n]['538CPCT__1'] = int(graph.nodes[n]['538CPCT__1'])
    graph.nodes[n]['538DEM_PL'] = int(graph.nodes[n]['538DEM_PL'])
    graph.nodes[n]['538GOP_PL'] = int(graph.nodes[n]['538GOP_PL'])
    graph.nodes[n]['8THGRADE_1'] = int(graph.nodes[n]['8THGRADE_1'])

#real partitions
partition_2011 = Partition(graph, "2011_PLA_1", updaters1)
partition_GOV = Partition(graph, "GOV", updaters1)
partition_TS = Partition(graph, "TS", updaters1)
partition_REMEDIAL = Partition(graph, "REMEDIAL_P", updaters1)
partition_CPCT = Partition(graph, "538CPCT__1", updaters1)
partition_DEM = Partition(graph, "538DEM_PL", updaters1)
partition_GOP = Partition(graph, "538GOP_PL", updaters1)
partition_8th = Partition(graph, "8THGRADE_1", updaters1)
partition_list = [
    partition_2011, partition_GOV, partition_TS, partition_REMEDIAL, partition_CPCT, partition_DEM, partition_GOP, partition_8th]
label_list = [
    '2011', 'GOV', 'TS', 'REMEDIAL', 'CPCT','DEM', 'GOP','8th']
ideal_population = sum(partition_2011["population"].values())/len(partition_2011)
print("Ideal population: ", ideal_population)

ccol = 'COUNTYFP10'

n_parts = []
c_bounds = []
shannon_entr = []
power_entr = []
n_split_locals = []
pa_fouls = []
symmetric_entr = []
n_pieces = []
for index, part in enumerate(partition_list):
    n_parts.append(num_parts(part, graph, ccol, gdf))
    c_bounds.append(coincident_boundaries(part, graph, ccol))
    shannon_entr.append(shannon_entropy(part, graph, ccol, gdf))
    power_entr.append(power_entropy(part, graph, ccol, gdf, 0.1))
    n_split_locals.append(num_split_localities(part, graph, ccol, gdf))
    pa_fouls.append(pennsylvania_fouls(part, graph, ccol, pop_col, gdf))
    symmetric_entr.append(symmetric_entropy(part, graph, ccol, pop_col, gdf))
    n_pieces.append(num_pieces(part, graph, ccol))

scores = [n_parts, c_bounds, shannon_entr, power_entr, n_split_locals, pa_fouls, symmetric_entr, n_pieces]
score_labels = ["number_of_parts", "coincident_boundaries_score", "shannon_entropy", "power_entropy", "number_of_split_localities", "Pennsylvania_fouls", "symmetric_entropy", "number_of_pieces"]
y_pos = np.arange(len(label_list))

for i in range (len(scores)):
    plt.figure()
    plt.bar(y_pos, scores[i], align='center', alpha=0.5)
    plt.xticks(y_pos, label_list)
    plt.ylabel(score_labels[i])
    plt.savefig(outdir + score_labels[i] + "_bar.png")
    plt.close()

for score_a, score_b in combinations(scores, 2):
    a_name = score_labels[scores.index(score_a)]
    b_name = score_labels[scores.index(score_b)]
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(score_a, score_b)

    minimum = min(score_a)
    maximum = max(score_a)

    x = np.linspace(minimum,maximum,100)
    y = slope * x + intercept

    r_label = 'R^2=' + str(r_value**2)

    plt.figure()
    plt.scatter(score_a, score_b)
    plt.plot(x, y, ':r', label=r_label)
    plt.xlabel(a_name)
    plt.ylabel(b_name)
    plt.legend(loc='upper left')
    plt.savefig(outdir + a_name + "_and_" + b_name + "_scatter.png")
    plt.close()

statistics = pd.DataFrame({
    "PLAN": label_list,
    "NUM_PARTS": n_parts,
    "COINCIDENT_BOUNDARIES": c_bounds,
    "SHANNON_ENTROPY": shannon_entr,
    "POWER_ENTROPY": power_entr,
    "NUM_SPLIT_LOCALITIES": n_split_locals,
    "PENNSYLVANIA_FOULS": pa_fouls,
    "SYMMETRIC_ENTROPY": symmetric_entr,
    "NUM_PIECES": n_pieces
})

statistics.to_csv(outdir + "pa_score_statistics.csv", index=False)