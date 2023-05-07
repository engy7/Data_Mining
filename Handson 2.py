import numpy as np
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('/Users/ENGY/Desktop/ASU/Senior 2/Data Mining/Dataset/store_data.csv', header=None)
print(store_data)

records_list= store_data.values.tolist()
clean_list = [[x for x in sublist if str(x) != 'nan' ] for sublist in records_list]
print (clean_list)


association_rules = apriori(clean_list, min_support=0.005, min_confidence=0.2, min_length=1)
association_results = list(association_rules)

print("There are {} Relation derived.".format(len(association_results)))


for item in association_results:

    print(item)
    pair = item[0]
    items = [x for x in pair]
    print("Frequent item sets: " + str(items))
    print("Support: " + str(item[1]))
    if (len(pair) > 1):
        for rule in item[2]:
            print("Rule: " + str(rule[0]) + "->" + str(rule[1]))
            print("Confidence: " + str(rule[2]))
            print("Lift: " + str(rule[3]))
            print("===========================================")
