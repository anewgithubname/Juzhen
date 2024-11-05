
#pragma once

#include <iomanip>
#include <iostream>
#include <algorithm>

// Function to plot a histogram on the console
void plot_histogram(const float *data, int size, int bins = 10)
{
    if (size == 0)
    {
        std::cout << "No data provided.\n";
        return;
    }

    // Find the min and max values in the data
    float min_val = *std::min_element(data, data + size);
    float max_val = *std::max_element(data, data + size);

    // Calculate the range and bin width
    float range = max_val - min_val;
    float bin_width = range / bins;

    // Initialize bins
    // int bin_counts[bins] = {0};
    int *bin_counts = new int[bins]();

    // Fill bins with frequency counts
    for (int i = 0; i < size; ++i)
    {
        int bin_index = std::min(static_cast<int>((data[i] - min_val) / bin_width), bins - 1);
        bin_counts[bin_index]++;
    }

    // Find the maximum count for scaling
    int max_count = *std::max_element(bin_counts, bin_counts + bins);

    // Calculate the necessary width for bin labels
    int label_width = std::max(std::to_string(static_cast<int>(std::ceil(max_val))).length(),
                               std::to_string(static_cast<int>(std::floor(min_val))).length()) +
                      4; // Extra for precision

    // Plot the histogram
    std::cout << "Histogram:\n";
    for (int i = 0; i < bins; ++i)
    {
        // Calculate bin range for display
        float bin_start = min_val + i * bin_width;
        float bin_end = bin_start + bin_width;

        // Print bin range with consistent width
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[" << std::setw(label_width) << bin_start << ", " << std::setw(label_width) << bin_end << "): ";

        // Print asterisks proportional to the frequency count
        int bar_length = static_cast<int>(50.0 * bin_counts[i] / max_count);
        std::cout << std::string(bar_length, '*') << " (" << bin_counts[i] << ")\n";
    }
    
    delete[] bin_counts;
}

// Function to display a progress bar in the console
void display_progress_bar(float progress, const char *msg = "")
{
    int bar_width = 50; // Width of the progress bar
    std::cout << "\r[";
    int pos = static_cast<int>(bar_width * progress);

    // Draw the progress bar
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            std::cout << "="; // Filled part
        else if (i == pos)
            std::cout << ">"; // Current position
        else
            std::cout << " "; // Empty part
    }

    // Display the progress percentage and the message
    std::cout << "] " << std::fixed << std::setprecision(0) << (progress * 100.0) << "% " << msg << "    \r";
    std::cout.flush(); // Flush to update the console without a new line
}