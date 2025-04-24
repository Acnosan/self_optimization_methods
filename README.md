# optimization_methods
Multiple Optimization Methods to find the optimal value(s) that minimizes a function.
Currently works for 2 variables functions (x,y)

## Included Methods:
| Methods | Algorithms |
| :---: | :---: |
| `Descend Methods` | Basic Gradient Descend <br> Newton |
| `Exact Line Search Methods` | Bi Section <br> Golden Section |
| `InExact Line Search Methods` | Armijo |

# How to Run :
To run the function you need to input this command in the terminal:
```python -m <method_folder>.<script_file>``` , for example : ```python -m descend_method.grad_desc```

# Requirements :
all the requirements to run the scripts are listed inside requirements.txt file,
To install the requirements input this command in the terminal:
```pip install -r requirements.txt```