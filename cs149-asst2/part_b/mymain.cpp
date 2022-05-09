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
int MyRunner::val = 0;
int main()
{
    TaskSystemParallelThreadPoolSleeping t(4);
    MyRunner runner;
    int A = t.runAsyncWithDeps(&runner, 2, std::vector<int>{});
    int B = t.runAsyncWithDeps(&runner, 2, std::vector<int>{A});
    t.runAsyncWithDeps(&runner, 2, std::vector<int>{B});
    t.sync();
}