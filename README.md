# Description:
The matlab and CUDA implementation of a O(mn^2) information flow analysis (the original algorithm is O(n^4)) algorithm (http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000350).

# Running Instructions
The input should be named "connection_matrix" and arranged in an n*n matrix
The matlab codes can be directly run on this file
For the cuda code, you need to run the preprocessor first to generate the input for the cuda code and also the parameters needed for the cuda code. Then change the parameters in the top of the cuda code according to the outputs of the preprocessor. Then make using make.bash and run.

# Time Test:
A network with N (number of nodes) = 3,447 and M (number of edges) = 92,026

| Algorithm | Runtime (s) | Speed Compared to Original |
| ----------- | ------------ | -------------- | 
| Original Algorithm (CPU, 4 cores)      | 20976.2      | 1.00x  |
| New Algorithm (CPU, 4 cores)       | 9433.4       | 2.22x |
| New Algorithm (GPU, GTX580x1)           | 1205.8       | 17.40x |

