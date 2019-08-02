
def vtds_per_district(locality_splits):
    """
    A function that counts the number of VTDs per district.
    
    :param county_splits: A dictionary with keys as district numbers and values Counter() dictionaries. The Counter dictionaries have pairs COUNTY_ID: NUM which counts the number of VTDS in the county in the district.
        
    :return: The total number of vtds in a district.
    """
    
    vtds = {}
    
    for district in locality_splits.keys():
        sum = 0
        counter = locality_splits[district]
        for vtd in counter.values():
            sum += vtd
        vtds[district] = sum
    return vtds 

def shannon_entropy(partition, locality_col, district):
    locality_splits, localities = locality_splits(partition, locality_col)
    vtds = vtds_per_district(locality_splits)

    total_vtds = 0
    for k,v in locality_splits.items():
        for x in list(v.values()):
            total_vtds += x
    
    some_vtds = locality_splits[district][locality_col]

    entropy = 0
    for locality_j in localities:
        inner_sum = 0
        q = some_vtds / total_vtds
        for district_i in range(num_districts):

            counter = locality_splits[district_i+1]
            intersection = counter[str(locality_j)]
            p = intersection / vtds[str(locality_j)]

            if p != 0:
                inner_sum += p * math.log(1/p)
        entropy += q * (inner_sum)
        #print(1/q * (inner_sum-1))
    return entropy

def power_entropy(partition, locality_col, alpha):
    """
    :param county_splits: a dictionary COUNTY : NUM OF SPLITS
    :param vtds: a dictionary of COUNTY: VTDS 
    :param total: a population count
    :param alpha: a value
        
    return entropy: an entropy score value
    """

    locality_splits, localities = locality_splits_dict(partition, locality_col)
    vtds = vtds_per_district(locality_splits)

    total_vtds = 0
    for k,v in locality_splits.items():
        for x in list(v.values()):
            total_vtds += x
    
    some_vtds = locality_splits[district][locality_col]
    
    entropy = 0
    for locality in localities:
        inner_sum = 0
        q = some_vtds / total_vtds
        for district_i in range(num_districts):
            counter = locality_splits[district_i+1]
            intersection = counter[str(locality_j)]
            p = intersection / vtds[str(locality_j)]

            inner_sum += p ** (1-alpha)
        entropy += 1/q * (inner_sum-1)
        #print(1/q * (inner_sum-1))
    return entropy


def district_splits_dict(locality_splits, localities):
    '''
    Args:
        county_splits: a dictionary with keys as district numbers and values Counter() dictionaries
                        these counter dictionaries have pairs COUNTY_ID : NUM which counts the number of VTDS
                        in the county in the district 
        
    Returns:
       district_splits: a dictionary that has as keys the county id and returns as values the
    districts in that county. 
    '''
    district_splits = {k:[] for k in localities}
    
    for locality in localities:
        districts = {}
        for district in locality_splits.keys():
            if locality in locality_splits[district].keys():
                district_splits[locality].append(district)
    return district_splits   

def pieces_allowed(localities, graph, to_add=1):
    district_splits ={}
    
    for county in counties:
        sg=graph.subgraph(n for n, v in graph.nodes(data=True) if v[county_col]==county)
        pop = 0
        
        for n in sg.nodes():
            pop += sg.node[n]["TOT_POP"]
        
        district_splits[county] = math.ceil(pop/(totpop/num_districts)) + to_add
    return district_splits

def pennsylvania_fouls(partition, graph, to_add):
    locality_splits, localities = locality_splits_dict(partition, locality_col)
    district_splits = district_splits_dict(locality_splits, localities)
    pieces = pieces_allowed(localities, graph, to_add)
    #vtds = vtds_per_district(locality_splits)

    too_many = 0
    for locality in localities:
        if len(district_splits[locality]) > pieces[locality]:
            too_many += 1
    
    return too_many
