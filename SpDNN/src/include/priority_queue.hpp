#ifndef SPDNN_PRIORITY_QUEUE
#define SPDNN_PRIORITY_QUEUE

#include <vector>
#include <map>
#include <tuple>

// #include <pybind11/pybind11.h>
// #include <pybind11/eigen.h>
// #include <pybind11/stl.h>
// #include <pybind11/iostream.h>


template <typename K, typename V>
class QueueEntry {
    public:
    QueueEntry<K, V>* prev;
    QueueEntry<K, V>* next;
    K key;
    K update;
    V value;

    QueueEntry(K key, K update, V value) 
        :key(key), update(update), value(value), prev(nullptr), next(nullptr) {}
};

template <typename K, typename V>
class HeadEntry {
    public:
    K key;
    QueueEntry<K, V>* head;
    QueueEntry<K, V>* end;

    HeadEntry(K key, QueueEntry<K, V>* head) : key(key), head(head), end(head) {}
};

template <typename K, typename V>
class PriorityQueue
{
    public:
        PriorityQueue(std::vector<std::tuple<K, V>> elements, K zero_elem);
        int decKey(K key, K decrement);
        int incKey(K key, K increment);
        V pop();

    private:
        void insertIntoQueue(QueueEntry<K, V>* v);
        std::map<K, HeadEntry<K, V>> head_table;
        std::vector<QueueEntry<K, V>> queue;
        std::map<V, QueueEntry<K, V>*> entries;
        QueueEntry<K, V>* top;
        K zero_elem;
};

// Do I need this?
#include "../priority_queue.cpp"

#endif // SPDNN_PRIORITY_QUEUE