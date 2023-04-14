import pandas as pd

# read in the expression parts of a count file
def readCountFile(fn):
    f = open(fn, 'r')
    gene_count_dict = {}
    for line in f:
        sl = line.split()
        if sl[0][:4] == 'ENSG':
            gene_count_dict[sl[0]] = sl[1]

    f.close()
    return gene_count_dict

# collect all keys in a dict of dict
def getAllKeys(dict_of_dict):
    all_keys = []
    for k in dict_of_dict.keys():
        all_keys.extend(list(dict_of_dict[k].keys()))

    return sorted(list(set(all_keys)))

# merge all count collections into a matrix
def mergeCountMat(gene_count_collection):
    all_keys = getAllKeys(gene_count_collection)

    count_dict = {}
    for sn in sample_name_arr:
        gene_count_dict = gene_count_collection[sn]
        count_arr = []
        for gn in all_keys:
            if gn in gene_count_dict.keys():
                num_count = gene_count_dict[gn]
            else:
                num_count = 'NaN'

            count_arr.append(num_count)

        count_dict[sn] = count_arr

    count_table = pd.DataFrame.from_dict(count_dict)
    count_table.index = all_keys

    return count_table


sample_name_arr = ["02-E1A", "293T-01-B7A", "HEK293-01-B7A"]

# read in the count files
gene_count_collection = {}
for sn in sample_name_arr:
    gene_count_collection[sn] = readCountFile(sn + '.counts')

# merge into table
count_table = mergeCountMat(gene_count_collection)
count_table.to_csv("collected_count_table.csv")
