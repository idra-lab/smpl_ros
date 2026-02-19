#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <cmath>

// Struttura dati per accumulare punti in un voxel
struct VoxelData {
    Eigen::Vector3d sum_pos   = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_norm  = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_color = Eigen::Vector3d::Zero();
    int count = 0;
};

// Chiave voxel robusta
struct VoxelKey {
    int x, y, z;
    VoxelKey(int xi, int yi, int zi) : x(xi), y(yi), z(zi) {}
};

// Hash combinato sicuro per VoxelKey
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey &k) const noexcept {
        std::size_t hx = std::hash<int>()(k.x);
        std::size_t hy = std::hash<int>()(k.y);
        std::size_t hz = std::hash<int>()(k.z);
        // combinazione con mix per ridurre collisioni
        return hx ^ (hy * 0x9e3779b97f4a7c15ULL) ^ (hz * 0xc6a4a7935bd1e995ULL);
    }
};

// Operatore di uguaglianza per unordered_map
struct VoxelKeyEq {
    bool operator()(const VoxelKey &a, const VoxelKey &b) const noexcept {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
};

// Funzione di voxel downsampling
std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>
voxelDownsample(
    const std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> &points,
    double voxel_size)
{
    std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash, VoxelKeyEq> voxels;

    for (const auto &pt_tuple : points) {
        const Eigen::Vector3d &pos   = std::get<0>(pt_tuple);
        const Eigen::Vector3d &color = std::get<1>(pt_tuple);
        const Eigen::Vector3d &norm  = std::get<2>(pt_tuple);

        // Indici voxel discreti
        int xi = static_cast<int>(std::floor(pos.x() / voxel_size));
        int yi = static_cast<int>(std::floor(pos.y() / voxel_size));
        int zi = static_cast<int>(std::floor(pos.z() / voxel_size));

        VoxelKey key(xi, yi, zi);
        auto &v = voxels[key];
        v.sum_pos   += pos;
        v.sum_norm  += norm;
        v.sum_color += color;
        v.count += 1;
    }

    // Costruzione point cloud downsampled
    std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> downsampled;
    downsampled.reserve(voxels.size());

    for (auto &kv : voxels) {
        const VoxelData &v = kv.second;

        Eigen::Vector3d avg_pos   = v.sum_pos / v.count;
        Eigen::Vector3d avg_color = v.sum_color / v.count;
        Eigen::Vector3d avg_norm  = v.sum_norm / v.count;

        // Normalizza le normali
        if (avg_norm.norm() > 1e-8) avg_norm.normalize();

        downsampled.emplace_back(avg_pos, avg_color, avg_norm);
    }

    return downsampled;
}
