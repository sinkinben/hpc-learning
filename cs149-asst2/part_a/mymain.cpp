#include "itasksys.h"
#include "tasksys.h"
#include <iostream>
class MyRunner : public IRunnable
{
public:
    void runTask(int i, int n)
    {
        printf("i = %d\n", i);
    }
    ~MyRunner() {}
};
int main()
{
    TaskSystemParallelSpawn spawn(4);
    MyRunner runner;
    spawn.run(&runner, 8);
}