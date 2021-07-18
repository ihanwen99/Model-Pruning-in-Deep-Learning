import json
index_file = {}
f = open('knn_res_hanwen_7.txt')
for j in f.readlines():
    filter = int(j.split('_')[1])
    label = int(j.split()[1])
    if label not in index_file:
        index_file[label] = []
    index_file[label].append(filter)
f.close()
f = open('filter_means_hanwen_7.json', 'w')
json.dump(index_file,f)
for k, v in index_file.items():
    print(k ,v)
#
# f = open('yils_v3.json')
# p = json.load(f)
# cluster_pres = {}
# for cluster_index, cluster in p.items():
#     cluster_pres[cluster_index] = {}
#     for i in range(10):
#         cluster_pres[cluster_index][i] = 0
#     for filter, pres in cluster.items():
#         for label, pre in pres.items():
#             if label == 'prefer':
#                 continue
#             cluster_pres[cluster_index][int(label)] += pre
#     cluster_pres[cluster_index]['prefer'] = \
#         max(cluster_pres[cluster_index], \
#             key=lambda k:cluster_pres[cluster_index][k])
# print(cluster_pres)
