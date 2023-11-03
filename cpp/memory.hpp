/**
 * @file memory.hpp
 * @brief Memory Control Module
 * @author Song Liu (song.liu@bristol.ac.uk)
 *
 * This file contains all memory related operations.
 *
    Copyright (C) 2023 Song Liu (song.liu@bristol.ac.uk)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

 */

#pragma once
#include "helper.hpp"

template <class D>
struct MemorySpace {
    D *ptr;
    size_t size;
};

template <class D>
class Memory {
    static std::list<MemorySpace<D>> alive_mems;
    static std::list<MemorySpace<D>> dead_mems;
    
    static D* _alloc(size_t size);
    static void _free(D* ptr);

   public:
    /**
    * @brief Check if there is memory in the dead memory list, 
    * - if so, return the pointer to the memory.
    * - if not, allocate new memory.
	* @param size size of the memory to be allocated
	* @return pointer to the allocated memory
    */
    static D *allocate(size_t size) {
        STATIC_TIC;

        // search for space in the freed space.
        auto it =
            std::find_if(dead_mems.begin(), dead_mems.end(),
                         [&](MemorySpace<D> mem) { return mem.size == size; });

        if (it != dead_mems.end()) {
            LOG_DEBUG("Found {} space in dead memory: {}, address: {}",
                      datatype(D), size * sizeof(D), fmt::ptr(it->ptr));
            alive_mems.push_back({it->ptr, it->size});
            D *ret = it->ptr;
            dead_mems.erase(it);
            STATIC_TOC;
            return ret;
        }

        // no space available, allocate new space
        D *ptr = NULL;
        try {
            ptr = _alloc(size);
        } catch (std::bad_alloc &e) {
            LOG_ERROR("Failed to allocate memory: {}", e.what());
            ERROR_OUT;
        }
        LOG_DEBUG("No {} space available, allocate new space: {}, address: {}",
                  datatype(D), size * sizeof(D), fmt::ptr(ptr));
        alive_mems.push_back({ptr, size});
        STATIC_TOC;
        return ptr;
    }

    /**
    * @brief add memory ptr to the dead memory list.
    * @param ptr pointer to the memory no longer in use
    */
    static void free(D *ptr) {
        STATIC_TIC;
        size_t size = 0;

        auto it =
            std::find_if(alive_mems.begin(), alive_mems.end(), [&](MemorySpace<D> mem) { return mem.ptr == ptr; });

        if (it != alive_mems.end()) {
            size = it->size;
            alive_mems.erase(it);
        } else {
            LOG_ERROR(
                "Failed to find address {} in alive memory type: {}", fmt::ptr(ptr), datatype(D));
            ERROR_OUT;
        }

        dead_mems.push_back({ptr, size});
        STATIC_TOC;
    }

    /**
    * @brief release all memory
    * @note this function is called when the program exits
    */
    ~Memory() {
        size_t size = 0;
        // for each loop
        for (MemorySpace<D> mem_i : alive_mems) {
            _free(mem_i.ptr);
            size += mem_i.size;
        }

        // for each loop
        for (MemorySpace<D> mem_i : dead_mems) {
            _free(mem_i.ptr);
            size += mem_i.size;
        }

        alive_mems.clear();
        dead_mems.clear();
        LOG_INFO("Total {} memory released: {:.2f} MB.", datatype(D),
                 size * sizeof(D) / 1024.0 / 1024.0);
    }
};

/**
* allocate host memory
*/
template <class D>
D* Memory<D>::_alloc(size_t size) {
    // if size = 0, allocate 1 element, this is to avoid error when size = 0
    if(size == 0) size = 1;
    return new D[size];
}

/**
* free host memory
*/
template <class D>
void Memory<D>::_free(D* ptr) {
    delete[] ptr;
}

template <class D>
std::list<MemorySpace<D>> Memory<D>::alive_mems;
template <class D>
std::list<MemorySpace<D>> Memory<D>::dead_mems;

