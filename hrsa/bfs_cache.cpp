#include "bfs_cache.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/stat.h>

namespace bfs_cache {

static constexpr std::size_t kMatBytes = sizeof(CMat3); // 144 bytes
static constexpr std::size_t kHeaderBytes = sizeof(std::uint64_t); // 8 bytes
static constexpr std::size_t kWarnBytes = static_cast<std::size_t>(5) * 1024ULL * 1024ULL * 1024ULL; // 5 GB

bool exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool save_mats(const std::string& path, const std::vector<CMat3>& mats) {
    const std::uint64_t count = static_cast<std::uint64_t>(mats.size());
    const std::size_t payload_bytes = static_cast<std::size_t>(count) * kMatBytes;
    const std::size_t total_bytes = kHeaderBytes + payload_bytes;

    if (total_bytes > kWarnBytes) {
        std::cerr << "[bfs_cache] WARNING: writing " << total_bytes
                  << " bytes (~" << (total_bytes >> 30) << " GB) to " << path
                  << std::endl;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[bfs_cache] save_mats: failed to open " << path
                  << " for writing" << std::endl;
        return false;
    }

    // Write little-endian uint64 count via memcpy of the local variable.
    // (No portability across architectures is required; this is a single-machine cache.)
    char header[kHeaderBytes];
    std::memcpy(header, &count, kHeaderBytes);
    out.write(header, kHeaderBytes);
    if (!out) {
        std::cerr << "[bfs_cache] save_mats: failed writing header to " << path
                  << std::endl;
        return false;
    }

    if (count > 0) {
        out.write(reinterpret_cast<const char*>(mats.data()),
                  static_cast<std::streamsize>(payload_bytes));
        if (!out) {
            std::cerr << "[bfs_cache] save_mats: failed writing payload to "
                      << path << std::endl;
            return false;
        }
    }

    out.close();
    if (!out) {
        std::cerr << "[bfs_cache] save_mats: failed closing " << path
                  << std::endl;
        return false;
    }
    return true;
}

bool load_mats(const std::string& path, std::vector<CMat3>& mats) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        std::cerr << "[bfs_cache] load_mats: stat failed for " << path
                  << std::endl;
        return false;
    }
    const std::size_t file_bytes = static_cast<std::size_t>(st.st_size);
    if (file_bytes < kHeaderBytes) {
        std::cerr << "[bfs_cache] load_mats: file too small (" << file_bytes
                  << " bytes) for header at " << path << std::endl;
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[bfs_cache] load_mats: failed to open " << path
                  << " for reading" << std::endl;
        return false;
    }

    char header[kHeaderBytes];
    in.read(header, kHeaderBytes);
    if (!in) {
        std::cerr << "[bfs_cache] load_mats: failed reading header from "
                  << path << std::endl;
        return false;
    }
    std::uint64_t count = 0;
    std::memcpy(&count, header, kHeaderBytes);

    const std::size_t expected = kHeaderBytes
        + static_cast<std::size_t>(count) * kMatBytes;
    if (file_bytes != expected) {
        std::cerr << "[bfs_cache] load_mats: size mismatch in " << path
                  << " (file=" << file_bytes << " expected=" << expected
                  << " count=" << count << ")" << std::endl;
        return false;
    }

    mats.resize(static_cast<std::size_t>(count));
    if (count > 0) {
        in.read(reinterpret_cast<char*>(mats.data()),
                static_cast<std::streamsize>(count) * static_cast<std::streamsize>(kMatBytes));
        if (!in) {
            std::cerr << "[bfs_cache] load_mats: failed reading payload from "
                      << path << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace bfs_cache
