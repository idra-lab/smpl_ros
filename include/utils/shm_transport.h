#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>

// ─── tunables ────────────────────────────────────────────────────────────────
static constexpr int    SHM_MAX_CAMS       = 4;
static constexpr int    SHM_PC_SLOTS       = 3;        // ring-buffer depth per camera
static constexpr int    SHM_MAX_POINTS     = 500'000;  // per slot
static constexpr int    SHM_SMPL_SLOTS     = 4;

// SHM segment names (in /dev/shm)
static constexpr const char* SHM_PC_NAME      = "/zed_pointclouds";
static constexpr const char* SHM_SMPL_NAME    = "/zed_smpl";
static constexpr const char* SHM_CAMINFO_NAME = "/zed_caminfo";

// ─── point layout: x y z  r g b  nx ny nz  (9 × float32) ────────────────────
static constexpr int    SHM_POINT_FLOATS   = 9;
using SHMFloat = float;

// ─── Point cloud shared memory ───────────────────────────────────────────────
//
//  [ SHMPCHeader ]
//  [ SHMPCSlot[SHM_PC_SLOTS] ]  × SHM_MAX_CAMS
//  [ float data[SHM_MAX_POINTS * SHM_POINT_FLOATS] ]  × (SHM_MAX_CAMS * SHM_PC_SLOTS)
//
struct alignas(64) SHMPCSlot {
    std::atomic<uint32_t> state;       // 0=free  1=writing  2=ready
    uint32_t              num_points;
    uint64_t              timestamp_ns;
    uint32_t              cam_idx;
    uint32_t              frame_id;
    uint8_t               _pad[40];
};
static_assert(sizeof(SHMPCSlot) == 64, "SHMPCSlot must be 64 bytes");

struct alignas(64) SHMPCHeader {
    std::atomic<uint64_t> write_seq[SHM_MAX_CAMS];
    uint32_t              num_cams;
    uint32_t              max_points;
    uint32_t              point_floats;
    uint32_t              num_slots;
    uint8_t               _pad[20];
};
static_assert(sizeof(SHMPCHeader) == 128, "SHMPCHeader size mismatch");

inline constexpr size_t shm_pc_total_size() {
    return sizeof(SHMPCHeader)
         + sizeof(SHMPCSlot) * SHM_MAX_CAMS * SHM_PC_SLOTS
         + sizeof(SHMFloat)  * SHM_MAX_CAMS * SHM_PC_SLOTS * SHM_MAX_POINTS * SHM_POINT_FLOATS;
}
inline SHMPCHeader* shm_pc_header(void* base) {
    return reinterpret_cast<SHMPCHeader*>(base);
}
inline SHMPCSlot* shm_pc_slot(void* base, int cam, int slot) {
    auto* slots = reinterpret_cast<SHMPCSlot*>(
        static_cast<uint8_t*>(base) + sizeof(SHMPCHeader));
    return &slots[cam * SHM_PC_SLOTS + slot];
}
inline SHMFloat* shm_pc_data(void* base, int cam, int slot) {
    size_t data_offset = sizeof(SHMPCHeader)
                       + sizeof(SHMPCSlot) * SHM_MAX_CAMS * SHM_PC_SLOTS;
    size_t slot_idx    = static_cast<size_t>(cam) * SHM_PC_SLOTS + slot;
    auto*  data        = reinterpret_cast<SHMFloat*>(
        static_cast<uint8_t*>(base) + data_offset);
    return data + slot_idx * SHM_MAX_POINTS * SHM_POINT_FLOATS;
}

// ─── SMPL shared memory ──────────────────────────────────────────────────────
static constexpr int SHM_SMPL_POSE_PARAMS  = 72;
static constexpr int SHM_SMPL_SHAPE_PARAMS = 10;
static constexpr int SHM_SMPL_TRANS_PARAMS = 3;

struct alignas(64) SHMSMPLSlot {
    std::atomic<uint32_t> state;
    uint32_t              valid;
    uint64_t              timestamp_ns;
    uint32_t              frame_id;
    uint8_t               _pad[44];

    float pose[SHM_SMPL_POSE_PARAMS];
    float betas[SHM_SMPL_SHAPE_PARAMS];
    float trans[SHM_SMPL_TRANS_PARAMS];
    float global_orient[3];
};

struct alignas(64) SHMSMPLHeader {
    std::atomic<uint64_t> write_seq;
    uint32_t              num_slots;
    uint8_t               _pad[52];
};

inline constexpr size_t shm_smpl_total_size() {
    return sizeof(SHMSMPLHeader) + sizeof(SHMSMPLSlot) * SHM_SMPL_SLOTS;
}
inline SHMSMPLHeader* shm_smpl_header(void* base) {
    return reinterpret_cast<SHMSMPLHeader*>(base);
}
inline SHMSMPLSlot* shm_smpl_slot(void* base, int slot) {
    return reinterpret_cast<SHMSMPLSlot*>(
        static_cast<uint8_t*>(base) + sizeof(SHMSMPLHeader)) + slot;
}

// ─── Camera info shared memory ───────────────────────────────────────────────
//
//  Scritto una volta sola all'init, letto dal consumer quando vuole.
//  Estrinseca: T_cam_to_world (4×4 float64, row-major)
//  Intrinseca: fx fy cx cy width height distortion[5] (k1 k2 p1 p2 k3)
//
//  [ SHMCamInfoHeader ]
//  [ SHMCamInfoEntry  × SHM_MAX_CAMS ]
//
struct alignas(64) SHMCamInfoEntry {
    // ── intrinsics ──────────────────────────────────────────
    float    fx, fy;          // focal lengths  [px]
    float    cx, cy;          // principal point [px]
    float    dist[5];         // k1 k2 p1 p2 k3 (OpenCV order)
    uint32_t width;           // image width  [px]
    uint32_t height;          // image height [px]
    uint32_t serial_number;   // ZED serial number
    uint8_t  _pad_intr[16];   // padding to 64-byte boundary for intr block

    // ── extrinsics: T_cam_to_world, 4×4 double row-major ────
    // i.e. the pose of the camera in the world (ROS) frame
    double   T_cam_to_world[16];  // 128 bytes

    // total: 4+4+4+4+20+4+4+4+16 = 64 bytes intr, + 128 bytes extr = 192 bytes
};
static_assert(sizeof(SHMCamInfoEntry) == 192, "SHMCamInfoEntry size mismatch");

struct alignas(64) SHMCamInfoHeader {
    std::atomic<uint32_t> ready;      // set to 1 when all entries are written
    uint32_t              num_cams;
    uint8_t               _pad[56];   // pad to 64 bytes
};
static_assert(sizeof(SHMCamInfoHeader) == 64, "SHMCamInfoHeader size mismatch");

inline constexpr size_t shm_caminfo_total_size() {
    return sizeof(SHMCamInfoHeader) + sizeof(SHMCamInfoEntry) * SHM_MAX_CAMS;
}
inline SHMCamInfoHeader* shm_caminfo_header(void* base) {
    return reinterpret_cast<SHMCamInfoHeader*>(base);
}
inline SHMCamInfoEntry* shm_caminfo_entry(void* base, int cam) {
    return reinterpret_cast<SHMCamInfoEntry*>(
        static_cast<uint8_t*>(base) + sizeof(SHMCamInfoHeader)) + cam;
}