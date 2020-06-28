# OLD - MIGHT NOT WORK RIGHT NOW
import os
import os.path as path
import numpy as np
import re
from prettytable import PrettyTable
from functools import reduce

# TODO: Turn this into arg
folder_path = path.join("G:", "BA", "Data", "Out")

all_files = os.listdir(folder_path)
all_files_iter = iter(all_files)

counts = {}

for file_name in all_files_iter:
    cl = re.search("^\d+-\d+-(\w+).png$", file_name)
    if cl == None:
        continue
    # print("filename: %s" % file_name)
    # print("match: %s" % cl[1])
    label = cl[1]
    if label in counts:
        counts[label] = counts[label] + 1
    else:
        counts[label] = 1

arr = []
for (key, count) in counts.items():
    arr.append((key, count))

sorted_arr = sorted(arr, key=lambda x: x[1])

sorted_arr.reverse()
number_of_images = reduce((lambda acc, el: acc + el[1]), sorted_arr, 0)
print(f"Number of images: {number_of_images}")

# for (label, count) in sorted_arr:
#     print(f"{label}: {count}")

tb = PrettyTable(["Nr", "Klasse", "Anzahl", "Anteil"])
for (nr, (label, count)) in enumerate(sorted_arr):
    tb.add_row([nr+1, label, count, round(count / number_of_images*100, 2)])

print(tb)
