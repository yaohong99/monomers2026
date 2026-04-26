#ifndef SELM_PARMPARSE_H
#define SELM_PARMPARSE_H

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <iostream>
#include <type_traits>
#include <stdexcept>

namespace SELM{

class ParmParse {
private:
    std::unordered_map<std::string, std::string> data;

public:
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Remove comments from line
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }

            // Skip line if it's empty or contains only whitespace
            size_t start = line.find_first_not_of(" \t\n\r\f\v");
            if (start == std::string::npos) {
                continue;
            }
            // Trim leading whitespace using find_first_not_of
            // line = line.substr(start);

            // Extract key and value
            std::istringstream iss(line);
            std::string key, value;
            if (!(iss >> key)) {
                continue; // Skip lines that can't parse a key
            }
            std::getline(iss, value);
            // auto first_non_space = value.find_first_not_of(" \t\n\r\f\v");
            // value = (first_non_space == std::string::npos) ? "" : value.substr(first_non_space);

            data[key] = value;
        }

        file.close();
        return true;

    }

    template<typename T>
    void get(const std::string& key, T& value) const {
        auto it = data.find(key);
        if (it == data.end()) {
            throw std::runtime_error("Key not found: " + key);
        }

        std::istringstream iss(it->second);
        if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T, std::string>) {
            iss >> value;
        } else {
            static_assert(std::is_array_v<T>, "Only single values or arrays are supported");
            processArray(iss, value);
        }
    }
    
    template<typename T>
    bool query(const std::string& key, T& value) const {
        auto it = data.find(key);
        if (it == data.end()) {
            return 1;
        }

        std::istringstream iss(it->second);
        if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T, std::string>) {
            iss >> value;
        } else {
            static_assert(std::is_array_v<T>, "Only single values or arrays are supported");
            processArray(iss, value);
        }
		return 0;
    }

private:
    template<typename T, std::size_t N>
    void processArray(std::istringstream& iss, T (&array)[N]) const {
        for (std::size_t i = 0; i < N; ++i) {
            if (!(iss >> array[i])) {
                throw std::runtime_error("Not enough data to fill the array for key: " + std::string(typeid(T).name()));
            }
        }
    }
};

}

#endif
