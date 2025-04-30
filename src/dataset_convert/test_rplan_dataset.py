from datasets import load_from_disk

procthor = load_from_disk('datasets/rplan_converted')
print(procthor['train'][88])
print(procthor['train'][100])
print(procthor['train'][1000])
print(procthor['train'][10000])
print(procthor['train'][30000])
