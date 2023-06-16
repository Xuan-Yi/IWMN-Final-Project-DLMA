# Introduction-to-Wireless-and-Mobile-Networking-Final-Project (DLMA experiments)

## Run code

* Use **Tensorflow**.

* There are 2 options

1. ```sim_env.ipynb```
2. Use modules in ```/sim_tools```. Please refer to ```sample_main.py``` for basic usage.

Basically, there are almost no differences between the two versions.

## Rules about protocol modules

* Each module includes **SEVERAL nodes** that running the same protocol.
* At least include ```tic()``` and ```reset()``` functions.
* ```tic()``` should return a 1D ndarray carrying the actions of every nodes.
* Add additionl functions if it's necesary.

## References

1. [DLMA (offcial repo)](https://github.com/YidingYu/DLMA.git)
2. [Deep-Reinforcement Learning Multiple Access for Heterogeneous Wireless Networks (IEEE JSAC, VOL. 37, NO. 6, JUNE 2019)](https://ieeexplore.ieee.org/document/8665952)
