#include "include/go.h"

template <typename K, typename V>
PriorityQueue<K, V>::PriorityQueue(std::vector<std::tuple<K, V>> elements, K zero_elem)
    : top(nullptr), zero_elem(zero_elem)
{
    for(auto elem : elements) {
        QueueEntry<K, V> q(std::get<0>(elem), zero_elem, std::get<1>(elem));
        this->vector.push_back(q);
        QueueEntry<K, V>& entry = this->vector.back();
        this->entries.insert({entry.value, &entry})

        (*this).insertIntoQueue(&entry);
    }
}

template <typename K, typename V>
int PriorityQueue<K, V>::decKey(K key, K decrement)
{
    QueueEntry<K, V>* entry = this->entries[key];
    entry->update -= decrement;

}

template <typename K, typename V>
int PriorityQueue<K, V>::incKey(K key, K increment)
{
    QueueEntry<K, V>* entry = this->entries[key];
    entry->update += increment;
    if(entry->update > this->zero_elem) {
        // Update Key
        entry->update = this->zero_elem;
        entry->key += increment;

        // Delete from current linked list
        entry->prev->next = entry->next;
        entry->next->prev = entry->prev;

        (*this).insertIntoQueue(entry);
    }
}

template <typename K, typename V>
V PriorityQueue<K, V>::pop()
{
    QueueEntry<K, V>* v = this->top;
    while(this->top->update < this->zero_elem) {
        K prev_key = v->key;
        v->key += v->update;
        v->update = this->zero_elem;
        if(v->key <= v->next->key) {
            //Update Head Table
            HeadEntry<K, V>& h = this->head_table[v->key];
            if(v->next->key == prev_key) {
                this->head_table[prev_key].head = v->next;
            } else{
                this->head_table.erase(prev_key);
            }

            // Update top
            this->top = v->next;
            this->top->prev = nullptr;

            // Update the position of v
            (*this).insertIntoQueue(v);
        }
        v = this->top;
    }

    // Update Head
    HeadEntry<K, V>& h = this->head_table[v->key];
    if(v->next->key == v->key) {
        this->head_table[v->key].head = v->next;
    } else{
        this->head_table.erase(v->key);
    }

    // Update Top
    this->top = v->next;
    this->top->prev = nullptr;
    return v->value;
}

template <typename K, typename V>
void PriorityQueue<K, V>::insertIntoQueue(QueueEntry<K, V>* v)
{
    if(this->top == nullptr) {
        this->top = v;
        HeadEntry<K, V> h(v->key, v);
        this->head_table.insert({v->key, h});
        return;
    }

    if(v->key > top->key){
        HeadEntry<K, V> h(v->key, v);
        this->head_table.insert({v->key, h});
        v->next = this->top;
        this->top->prev = v;
        v->prev = nullptr;
        return;
    }

    QueueEntry<K, V>* prev;
    if (this->head_table.find(v->key) == this->head_table.end()) {
        // Insert into head table and find previous entry
        HeadEntry<K, V> h(v->key, v);
        this->head_table.insert({v->key, h});

        K minKey = top->key;
        prev = this->head_table[top->key].end;
        for(typename std::map<K,V>::iterator iter = this->head_table.begin(); \
                    iter != this->head_table.end(); ++iter)
        {
            K k =  iter->first;
            if(k < minKey && v->key < k){

                prev = iter->second.end;
            } 
        }
    } else {
        HeadEntry<K, V>& h = this->head_table[v->key];
        prev = h.end;
        h.end = v;
    }

    v->next = prev->next;
    v->prev = prev;
    v->next->prev = v;
    v->prev->next = v;    
}