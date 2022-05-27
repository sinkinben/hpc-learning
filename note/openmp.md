## Open Multi-processing

OMP 可以使多线程编程变得简单。

使用一些宏定义，即可并行化执行一个代码块。

```cpp
#prama omp parallel for
for (int i = 0; i < n; ++i)
{
    // ...
}
```

更多用法：https://www.math.hkbu.edu.hk/parallel/pgi/doc/pgiws_ug/pgi32u12.htm#Heading144
