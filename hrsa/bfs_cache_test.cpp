#include "bfs_cache.h"
#include <iostream>
#include <vector>
int main() {
    std::vector<CMat3> v(1000);
    for (int i = 0; i < 1000; ++i)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                v[i].m[a][b] = std::complex<double>(i*9 + a*3 + b, 0.5);

    bool ok1 = bfs_cache::save_mats("/tmp/bfs_cache_test.bin", v);
    std::cout << "save: " << ok1 << "\n";

    std::vector<CMat3> loaded;
    bool ok2 = bfs_cache::load_mats("/tmp/bfs_cache_test.bin", loaded);
    std::cout << "load: " << ok2 << "  size: " << loaded.size() << "\n";

    int n_diff = 0;
    for (size_t i = 0; i < v.size(); ++i)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                if (v[i].m[a][b] != loaded[i].m[a][b]) ++n_diff;
    std::cout << "diff entries: " << n_diff << "\n";
    return n_diff == 0 ? 0 : 1;
}
