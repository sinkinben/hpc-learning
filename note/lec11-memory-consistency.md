## Memory Consistency

虽然 Coherence 和 Consistency 都被翻译为「一致性」，但二者其实有不同的含义：

- Coherence 一般指的是多核 CPU 中的 Cache 的一致性（L1/L2 Cache 是每个核独有的）。
  - 强调的是**相同的内存位置**，在不同 Core-cache 中的一致性。
  - 「一致」指的是内存的副本的「一致」。
- Consistency 
  - Coherence only guarantees that writes to address X willeventually propagate to other processors
  - Consistency deals with when writes to X propagate to other processors, relative to reads and writes to other addresses
  - Consistency 强调的是**何时**把写入的结果，通知其他的线程/分布式节点/Core，保证不同的线程/节点在同一时间，看到内存的状态是「一致」的。
  - 根据「时机」的不同，有不同的协议。例如，顺序一致性、最终一致性、因果一致性等。





