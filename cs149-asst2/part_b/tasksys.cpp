#include "tasksys.h"
#include <algorithm>
#include <cassert>


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads)
: ITaskSystem(num_threads), stop(false), remained_tasks(0), workers(num_threads), task_id(0)
{
    auto worker = [this]() {
        while (1)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx_que);
                cv_new_task.wait(lock, [this]() { return stop.load() || !tasks.empty(); });
                if (stop.load() && tasks.empty())
                    return;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
            remained_tasks -= 1;
            if (remained_tasks.load() == 0)
                cv_all_tasks_done.notify_all();
        }
    };
    for (int i = 0; i < num_threads; ++i)
        workers[i] = std::thread(worker);
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping()
{
    stop = true;
    cv_new_task.notify_all();
    for (auto &th : workers)
        th.join();
}

void TaskSystemParallelThreadPoolSleeping::addBatchTasks(IRunnable *runnable, int num_tasks)
{
    std::unique_lock<std::mutex> lock(mtx_que);
    for (int i = 0; i < num_tasks; ++i)
    {
        tasks.emplace([=]() { runnable->runTask(i, num_tasks); });
        remained_tasks++;
        cv_new_task.notify_one();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable *runnable, int num_total_tasks)
{
    addBatchTasks(runnable, num_total_tasks);
    std::unique_lock<std::mutex> lock(mtx_all_tasks_done);
    cv_all_tasks_done.wait(lock, [this]() { return remained_tasks.load() == 0; } );
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks, const std::vector<TaskID>& deps)
{
    task_id += 1;
    indeg[task_id] = deps.size();
    tasks_collect[task_id] = AsyncTask(runnable, num_total_tasks);

    for (int x : deps)
        graph[x].emplace(task_id);

    if (deps.empty())
    {
        ready_que.emplace(task_id);
        indeg.erase(task_id);
    }
    return task_id;
}

void TaskSystemParallelThreadPoolSleeping::sync()
{
    TaskID id;
    while (!ready_que.empty())
    {
        id = ready_que.front(), ready_que.pop();

        AsyncTask task = tasks_collect[id];
        run(task.first, task.second);

        for (int next : graph[id])
        {
            assert(indeg.count(next));
            indeg[next]--;
            if (indeg[next] == 0)
            {
                ready_que.emplace(next);
                indeg.erase(next);
            }
        }
        graph.erase(id);
    }
    return;
}
