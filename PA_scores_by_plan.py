#import
import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election, metrics)
from gerrychain.proposals import recom
from gerrychain.updaters import cut_edges
from functools import partial
import pandas
import networkx as nx
import numpy as np
import geopandas as gpd
import shapely
from scipy.stats.stats import pearsonr

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
for index, part in enumerate(partition_list):
    n_parts.append(num_parts(part, graph, ccol, gdf))
    c_bounds.append(coincident_boundaries(part, graph, ccol))
    shannon_entr.append(shannon_entropy(part, graph, ccol, gdf))
    power_entr.append(power_entropy(part, graph, ccol, gdf, 0.1))
    n_split_locals.append(num_split_localities(part, graph, ccol, gdf))
    pa_fouls.append(pennsylvania_fouls(part, graph, ccol, pop_col, gdf))


csplits = []
for index, part in enumerate(partition_list):
    ci = county_intersections(part, ccol)
    csplits.append(sum([len(x) for x in ci.values()])-len(ci))
y_pos = np.arange(len(label_list))
plt.bar(y_pos, csplits, align='center', alpha=0.5)
plt.xticks(y_pos, label_list)
plt.ylabel('county splittings')

statistics = pd.DataFrame({
    "NUM_PARTS": n_parts,
    "COINCIDENT_BOUNDARIES": c_bounds,
    "SHANNON_ENTROPY": shannon_entr,
    "POWER_ENTROPY": power_entr,
    "NUM_SPLIT_LOCALITIES": num_split_localities,
    "PENNSYLVANIA_FOULS": pa_fouls
})

statistics.to_csv(outdir + "pa_score_statistics.csv", index=False)