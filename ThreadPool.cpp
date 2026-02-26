#include "ThreadPool.h"

ThreadPool::ThreadPool(size_t numThreads) : shouldStop(false) {
	for (size_t i = 0; i < numThreads; ++i) {
		workers.emplace_back([this]() {
			while (true) {
				std::function<void()> task;

				{
					std::unique_lock<std::mutex> lock(queueMutex);
					condition.wait(lock,
												 [this]() { return shouldStop || !tasks.empty(); });

					if (shouldStop)
						return;

					task = std::move(tasks.front());
					tasks.pop();

					++activeTasks;
				}

				task();

				{
					std::lock_guard<std::mutex> lock(queueMutex);
					--activeTasks;

					if (tasks.empty() && activeTasks == 0) {
						finishedCondition.notify_all();
					}
				}
			}
		});
	}
}

void ThreadPool::ThreadPool::wait() {
	std::unique_lock<std::mutex> lock(queueMutex);
	finishedCondition.wait(
			lock, [this]() { return tasks.empty() && activeTasks == 0; });
}

ThreadPool::~ThreadPool() {
	{
		std::lock_guard<std::mutex> lock(queueMutex);
		shouldStop = true;
	}

	condition.notify_all();

	for (std::thread &worker : workers)
		worker.join();
}
