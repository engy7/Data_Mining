import pandas as pd
import pyfpgrowth

dataFrame= pd.read_csv('/Users/ENGY/Desktop/ASU/Senior 2/Data Mining/Dataset/store_data.csv',header=None)
print(dataFrame)

records_list= dataFrame.values.tolist()
print (dataFrame)

clean_list = [[x for x in sublist if str(x) != 'nan' ] for sublist in records_list]
print (clean_list)

frequeuentPatterns= pyfpgrowth.find_frequent_patterns(transactions=clean_list,support_threshold=200)
print(frequeuentPatterns)

Rules=pyfpgrowth.generate_association_rules(patterns=frequeuentPatterns, confidence_threshold=0.2)
print(Rules)