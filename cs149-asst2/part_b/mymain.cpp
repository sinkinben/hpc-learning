#include "itasksys.h"
#include "tasksys.h"
#include <iostream>
class MyRunner : public IRunnable
{
public:
    static int val;
    void runTask(int i, int n)
    {
        printf("val = %d \n", ++val);
    }
    ~MyRunner() {}
};

class OtherRunner : public IRunnable
{
public:
    ~OtherRunner() {}
    void runTask(int i, int n)
    {
        printf("Other runner: %d\n", i);
    }
};
int MyRunner::val = 0;
int main()
{
    int n = 3;
    TaskSystemParallelThreadPoolSleeping t(4);
    MyRunner runner;
    OtherRunner OtherRunner;
    int A = t.runAsyncWithDeps(&runner, n, std::vector<int>{});
    int B = t.runAsyncWithDeps(&runner, n, std::vector<int>{A});
    t.runAsyncWithDeps(&OtherRunner, n, {});
    t.runAsyncWithDeps(&runner, n, std::vector<int>{B});
    t.sync();
}