/**
 * @file helper.cpp
 * @brief implemenetation of some helper functions. 
 * @author Song Liu (song.liu@bristol.ac.uk)
 * 
    Copyright (C) 2021 Song Liu (song.liu@bristol.ac.uk)

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

#include "helper.h"
#include <iostream>

typedef vector<int> idxlist;

int time_in_ms(Clock::time_point start, Clock::time_point end) {
    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
}

idxlist seq(int start, int end) {
    idxlist ret;
    for (int i = start; i < end; i++) {
        ret.push_back(i);
    }
    return ret;
}

idxlist seq(int end){
    return seq(0, end);
}

idxlist shuffle(int start, int end){
    idxlist ret = seq(start, end);
    random_device rd;
    mt19937 g(rd());
    shuffle(ret.begin(), ret.end(), g);
    return ret;
}

idxlist shuffle(int end){
    return shuffle(0, end);
}

int rand_number(){
    return rand() % 10;
}