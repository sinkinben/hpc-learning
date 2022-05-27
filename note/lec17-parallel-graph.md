## Parallel Programming on Graph

Graph Computing 面临的挑战：

- 图的规模过大，无法完整载入内存
- BFS/DFS 需要额外的空间记录信息，至少需要 $O(V)$  空间记录某个顶点是否被访问
- 并行化的图算法
  - 一般而言，平常使用的都是 Top-Down-BFS ，可并行性差。
  - Bottom-Up-BFS 可提高并行性。
  - 二者区别：Top-Down-BFS 每次迭代基于 outgoing-edge ，而 Bottom-Up-BFS 每次迭代基于 incoming-edge .
