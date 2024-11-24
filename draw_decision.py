import matplotlib
import numpy as np
import argparse 

parser = argparse.ArgumentParser(description='Draw decision curves')
parser.add_argument('base', type=float, help='log file')
parser.add_argument('file', type=str, help='log file')

args = parser.parse_args()
log_file = args.file
decision = []
conf = []
# read lines from log file
with open(log_file, 'r') as f:
    lines = f.readlines()

    for line in lines:
        if line.startswith("Decision: "):
            indices = line.split("[")[1].split("]")[0]
            indices = indices.split(", ")
            indices = list(map(int, indices))
            decision.append(indices)
        elif line.startswith("Confidence: "):
            conf_value = line.split("[")[1].split("]")[0]
            conf_value = conf_value.split(", ")
            conf_value = list(map(float, conf_value))
            if len(conf_value)!=len(indices):
                conf_value = np.repeat(conf_value, len(indices)//len(conf_value))
            conf.append(conf_value)
print(len(decision))
print(len(conf))

# draw decision curves
import matplotlib.pyplot as plt

curves = [ [] for _ in range(len(decision[0])) ]
confidence = [ [] for _ in range(len(decision[0])) ]
for d in decision:
    for i, idx in enumerate(d):
        curves[i].append(idx)
for c in conf:
    for i, idx in enumerate(c):
        confidence[i].append(idx)

plt.figure(figsize=(8, 4))

x = np.arange(len(curves[0])*100, step=100)
# highlights the decision with different alpha, the higher the confidence, the darker the color
# choose a better colormap for better visualization
# set y axis to all layers, with the step=1
plt.yticks(np.arange(28))
cmap = [
    "#2c3e50", "#34495e", "#1a5276", "#145a32", "#0e6251",
    "#4b3832", "#b03a2e", "#76448a", "#7d6608", "#b9770e",
    "#6e2c00", "#1b2631", "#117864", "#154360", "#283747",
    "#78281f", "#5b2c6f", "#873600", "#145a32", "#7e5109"
]

base = args.base
for i, curve in enumerate(curves):
    plt.plot(x, curve, label=f'Layer {i}', alpha=0.4, c=cmap[i])
    plt.scatter(x, curve, alpha=(np.array(confidence[i])-base) / (1-base), c=cmap[i], s=15)

# set alpha for x and y axis
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_alpha(0.0)

plt.xticks(fontsize=10)
plt.yticks(fontsize=8)
plt.xlabel('Train Iterations', fontsize=12)
plt.ylabel('Winner Layers', fontsize=12)

# draw heatbar, the transparency is the confidence

plt.savefig('decision.png', bbox_inches='tight', dpi=300)
plt.savefig('decision.pdf', bbox_inches='tight', dpi=300)

