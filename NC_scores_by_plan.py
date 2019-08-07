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

#make new directory
outdir = "./nc_score_outputs/"
try:
    os.mkdir(outdir)
    print("Directory " , outdir ,  " created ") 
except FileExistsError:
    print("Directory " , outdir ,  " already exists")
print()

#load the dual graph
graph = Graph.from_file("./notebooks/NC_VTD/NC_VTD.shp", ignore_errors=True)
gdf = gpd.read_file("./notebooks/NC_VTD/NC_VTD.shp")

pop_col = "TOTPOP"
ccol = "County"

#updaters
updaters1 = {
    "population": updaters.Tally("TOTPOP", alias="population"),
    "cut_edges": cut_edges,
}

#real partitions
judge_plan = Partition(graph, "judge", updaters1)
old_plan = Partition(graph, "oldplan", updaters1)
new_plan = Partition(graph, "newplan", updaters1)

partition_list = [judge_plan, old_plan, new_plan]
label_list = ["judge_plan", "old_plan", "new_plan"]

#compute scores
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


for i in range (len(partition_list)):
    part = partition_list[i]

    gdfc = gdf.dissolve(by=ccol)
    assignment_series = part.assignment.to_series()
    gdf["assignment"] = assignment_series
    axes = gdfc.boundary.plot(figsize=(20,15), color=None, edgecolor="black", linewidth=0.5)
    gdf.plot(column="assignment", cmap='tab20', ax=axes)
    plt.savefig(outdir + label_list[i] + "_map.png")
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