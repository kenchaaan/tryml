# 確認

```python
from cycle1 import checks as ck
# check 1
ck.show_normal_dist()

# check 2
ck.solve_2dim_equations(a=3, b=0, c=0, d=2, i=12, j=26)

# check 3
ck.out_normal_data_to_csv(filepath='./cycle1/resources/male.csv', num=1000, mu=172.14, sigma=5.57)

# check 4
ck.plot_data_from_csv(filepath='./cycle1/resources/male.csv')
```

# 実践

```python
# star
from cycle1 import practices as pr
import numpy as np

x = np.arange(0, 1, 0.01)
pr.plot_data(data=x)
pr.out_csv(data=x, file='./resources/sin.csv')

# extra
import numpy as np
import matplotlib.pyplot as plt
from cycle1 import practices as pr

generation = np.arange(0, 120, 1)
for j in np.arange(0.25 - 0.001, 0.25 + 0.001, 0.001):
    plt.plot(generation, pr.calculate_n_generation(1, generation, j), label=str(j))
plt.legend(loc="lower right")
plt.show()
```