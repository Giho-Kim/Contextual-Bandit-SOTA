# BRTS
Python implementation of the Balanced Linear Contextual Bandits (BRTS) [[1]](#1) with baseline method (LinTS) [[2]](#2), and Random policy.

## Usage
With the default setting, you can run experiment as follows:
```
python main.py
```
The parameters can be changed by adding additional command-line arguments:
```
python main.py --lam=0.1 --thres=0.5 --alpha=1.0 --T=1000 --fill_buffer=10
```


## Reference
<a id="1">[1]</a>
Dimakopoulou, M., Zhou, Z., Athey, S., & Imbens, G. (2019). Balanced Linear Contextual Bandits. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 3445-3453. https://doi.org/10.1609/aaai.v33i01.33013445

<a id="2">[2]</a>
Dimakopoulou, M., Zhou, Z., Athey, S., & Imbens, G. (2019). Balanced Linear Contextual Bandits. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 3445-3453. https://doi.org/10.1609/aaai.v33i01.33013445
