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
from functools import partial
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

shapefile = "https://github.com/mggg-states/PA-shapefiles/raw/master/PA/PA_VTD.zip"

df = gpd.read_file(shapefile)

#For Pennsylvania
county_col = "COUNTYFP10"
pop_col = "TOT_POP"
uid = "GEOID10"

graph = Graph.from_geodataframe(df,ignore_errors=True)
graph.add_data(df,list(df))
graph = nx.relabel_nodes(graph, df[uid])
print("Graph loaded.")

starting_partition = GeographicPartition(
    graph,
    assignment="2011_PLA_1",
    updaters={
        "polsby_popper" : polsby_popper,
        "cut_edges": cut_edges,
        "population": Tally(pop_col, alias="population"),
    }
)
print("Partition generated.")
print()

print("Number of parts:", num_parts(starting_partition, graph, county_col, df))
print("Coincident boundaries score:", coincident_boundaries(starting_partition, graph, county_col))
print("Shannon entropy:", shannon_entropy(starting_partition, graph, county_col, df))
print("Power entropy:", power_entropy(starting_partition, graph, county_col, df, 0.1))
print("Number of split localities:", num_split_localities(starting_partition, graph, county_col, df))
print("Pennsylvania fouls:", pennsylvania_fouls(starting_partition, graph, county_col, pop_col, df))
print("Symmetric entropy:", symmetric_entropy(starting_partition, graph, county_col, pop_col, df))
print("Number of pieces:", num_pieces(starting_partition, graph, county_col))