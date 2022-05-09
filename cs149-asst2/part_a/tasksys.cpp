#include "tasksys.h"
#include <algorithm>
#include <vector>
#include <thread>
#include <atomic>

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

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads)
: ITaskSystem(num_threads), num_threads(num_threads) {}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    std::vector<std::thread> pool;
    std::atomic_int32_t task_id(0);

    /* Concurrent threads to "eat" each id of tasks */
    auto worker = [=, &task_id]() {
        while (task_id.load() < num_total_tasks)
        {
            runnable->runTask(task_id++, num_total_tasks);
        }
    };
    /* Create n threads to "eat" */
    for (int i = 0; i < num_threads; ++i)
        pool.emplace_back(std::thread(worker));
    for (auto &th : pool)
        th.join();
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    return 0;
}

void TaskSystemParallelSpawn::sync() {
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

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads)
: ITaskSystem(num_threads), stop(false), remained_tasks(0), workers(num_threads)
{
    auto worker = [this]() {
        while (!stop)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lck(mtx);
                if (tasks.empty())
                    continue;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
            remained_tasks--;
        }
    };
    for (int i = 0; i < num_threads; ++i)
        workers[i] = std::thread(worker);
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() 
{
    stop = true;
    for (auto &th : workers)
        th.join();
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks)
{
    mtx.lock();
    for (int i = 0; i < num_total_tasks; ++i)
    {
        tasks.emplace([=]() { runnable->runTask(i, num_total_tasks); } );
        remained_tasks++;
    }
    mtx.unlock();

    while (remained_tasks.load() > 0)
        ;
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
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
: ITaskSystem(num_threads), stop(false), remained_tasks(0), workers(num_threads)
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

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks)
{
    {
        std::unique_lock<std::mutex> lock(mtx_que);
        for (int i = 0; i < num_total_tasks; ++i)
        {
            tasks.emplace([=]() { runnable->runTask(i, num_total_tasks); });
            remained_tasks++;
            cv_new_task.notify_one();
        }
    }
    std::unique_lock<std::mutex> lock(mtx_all_tasks_done);
    cv_all_tasks_done.wait(lock, [this]() { return remained_tasks.load() == 0; } );
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
