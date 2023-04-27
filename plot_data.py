import pandas
import matplotlib.pyplot as plt

data_path = r'C:\Program Files (x86)\Steam\steamapps\common\Farming Simulator 22\telemetry.csv'

data = pandas.read_csv(data_path)
# Columns are dt,vX,vY,vZ,gx,gz,moveForwards,maxSpeed

# Integrate the dt column.
t = data['dt'].cumsum()

# Plot all but the last two columns.
fig, ax = plt.subplots()
for col in data.columns[:-2]:
    # If col ends in x, y, or z, latexify it.
    label = str(col).lower()
    if label[-1] in 'xyz':
        label = '$%s_{%s}$' % (label[:-1], label[-1])
    elif label == 'dt':
        label = '$\Delta t$'
    ax.plot(t/1e6, data[col], label=label)
ax.set_xlabel(r'$10^{-6} \cdot \int \Delta t \mathrm{d}t$')
ax.legend()
fig.tight_layout()

fig.savefig(data_path + '.png', dpi=300)
# fig.savefig(data_path + '.pdf', dpi=300)

plt.show()
