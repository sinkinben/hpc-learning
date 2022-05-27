## Data Parallel Thinking

- This lecture will include these subjects: map, sort, filter, groupBy, fold/reduce, join, scan/segmented scan, partition/flatten.
- Main idea: high-performance parallel implementations of these operations exist. So programs written in terms of these primitives can often run efficiently on a parallel machine

## Map, Fold and Reduce

The map function should be **side-effect free**.

- Map: `std::transform`
- Fold: `std::accumulate`
- Reduce: `std::reduce`



## Inclusive and Exclusive Scan

Inclusive/exclusive 表示是否包括 `nums[i]` 的前缀和（默认使用 `std::plus<>` 作为操作符）。

- `std::exclusive_scan`
- `std::includesive_scan`

https://en.cppreference.com/w/cpp/algorithm/inclusive_scan

串行算法是 `O(N)` 的，并行的 `O(1.5 * N)` 算法是重点。