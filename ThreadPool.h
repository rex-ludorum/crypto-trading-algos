#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
	explicit ThreadPool(size_t numThreads);
	~ThreadPool();

	// Must stay in header because it's a template
	template <class F> void enqueue(F &&task);
	void wait();

private:
	std::vector<std::thread> workers;
	std::queue<std::function<void()>> tasks;

	std::mutex queueMutex;
	std::condition_variable condition, finishedCondition;
	std::atomic<bool> shouldStop;

	size_t activeTasks = 0;
};

//
// Template implementation MUST be in header
//
template <class F> void ThreadPool::enqueue(F &&task) {
	{
		std::lock_guard<std::mutex> lock(queueMutex);
		tasks.emplace(std::forward<F>(task));
	}
	condition.notify_one();
}

#endif
