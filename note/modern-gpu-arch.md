## Modern GPU



## Intro

- Devoting more transistors to data processing, e.g., floating-point computations, is beneficial for highly parallel computations;
- the GPU can hide memory access latencies with computation, instead of relying on large data caches and complex flow control to avoid long memory access latencies, both of which are expensive in terms of transistors.

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/gpu-devotes-more-transistors-to-data-processing.png" style="width:80%" />



## Programming Model

At its core are three key abstractions:

- a hierarchy of thread groups, 
- shared memories, and 
- barrier synchronization - that are simply exposed to the programmer as a minimal set of language extensions.





