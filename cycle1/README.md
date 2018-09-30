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
