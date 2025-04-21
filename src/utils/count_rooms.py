from datasets import load_from_disk
import matplotlib.pyplot as plt
from collections import Counter

dataset = load_from_disk("datasets/rplan_converted")

room_counts = dataset["train"]["room_count"]

count_dict = Counter(room_counts)
print(count_dict)

# unique_counts = sorted(count_dict.keys())
# frequencies = [count_dict[val] for val in unique_counts]

# plt.figure(figsize=(10, 6))
# plt.bar(unique_counts, frequencies, edgecolor="black")

# plt.xlabel("Room Count")
# plt.ylabel("Frequency")
# plt.title("Room Count")
# plt.show()
