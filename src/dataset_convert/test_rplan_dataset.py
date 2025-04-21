from datasets import load_dataset, load_from_disk

procthor = load_from_disk('datasets/rplan_converted')
print(procthor['train'][0])
print(procthor['train'][10])
print(procthor['train'][88])
print(procthor['train'][100])
print(procthor['train'][1000])
