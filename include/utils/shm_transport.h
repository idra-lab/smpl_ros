#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>

// Shared memory layout definitions for ZED SMPL tracking -> to avoid ROS2
// overhead ─── tunables
// ────────────────────────────────────────────────────────────────
static constexpr int SHM_MAX_CAMS = 4;
static constexpr int SHM_PC_SLOTS = 3;         // ring-buffer depth per camera
static constexpr int SHM_MAX_POINTS = 500'000; // per slot
static constexpr int SHM_SMPL_SLOTS = 4;

// SHM segment names (in /dev/shm)
static constexpr const char *SHM_PC_NAME = "/zed_pointclouds";
static constexpr const char *SHM_SMPL_NAME = "/zed_smpl";

// ─── point layout: x y z  r g b  nx ny nz  (9 × float32) ────────────────────
static constexpr int SHM_POINT_FLOATS = 9; // x y z r g b nx ny nz
using SHMFloat = float;

// ─── Point cloud shared memory ───────────────────────────────────────────────
//
//  [ SHMPCHeader ]
//  [ SHMPCSlot[SHM_PC_SLOTS] ]  × SHM_MAX_CAMS
//  [ float data[SHM_MAX_POINTS * SHM_POINT_FLOATS] ]  × (SHM_MAX_CAMS *
//  SHM_PC_SLOTS)
//
struct alignas(64) SHMPCSlot {
  std::atomic<uint32_t> state; // 0=free  1=writing  2=ready
  uint32_t num_points;
  uint64_t timestamp_ns;
  uint32_t cam_idx;
  uint32_t frame_id;
  uint8_t _pad[40];
  // data follows in the data section (see SHMPCLayout)
};
static_assert(sizeof(SHMPCSlot) == 64, "SHMPCSlot must be 64 bytes");

struct alignas(64) SHMPCHeader {
  std::atomic<uint64_t> write_seq[SHM_MAX_CAMS]; // bumped after each write
  uint32_t num_cams;
  uint32_t max_points;
  uint32_t point_floats; // floats per point (9)
  uint32_t num_slots;    // slots per camera (SHM_PC_SLOTS)
  uint8_t _pad[20];
};
static_assert(sizeof(SHMPCHeader) == 128, "SHMPCHeader size mismatch");

// Helper: total size of the PC segment
inline constexpr size_t shm_pc_total_size() {
  return sizeof(SHMPCHeader) + sizeof(SHMPCSlot) * SHM_MAX_CAMS * SHM_PC_SLOTS +
         sizeof(SHMFloat) * SHM_MAX_CAMS * SHM_PC_SLOTS * SHM_MAX_POINTS *
             SHM_POINT_FLOATS;
}

// Pointer helpers (call after mmap)
inline SHMPCHeader *shm_pc_header(void *base) {
  return reinterpret_cast<SHMPCHeader *>(base);
}
inline SHMPCSlot *shm_pc_slot(void *base, int cam, int slot) {
  auto *slots = reinterpret_cast<SHMPCSlot *>(static_cast<uint8_t *>(base) +
                                              sizeof(SHMPCHeader));
  return &slots[cam * SHM_PC_SLOTS + slot];
}
inline SHMFloat *shm_pc_data(void *base, int cam, int slot) {
  size_t data_offset =
      sizeof(SHMPCHeader) + sizeof(SHMPCSlot) * SHM_MAX_CAMS * SHM_PC_SLOTS;
  size_t slot_idx = static_cast<size_t>(cam) * SHM_PC_SLOTS + slot;
  auto *data =
      reinterpret_cast<SHMFloat *>(static_cast<uint8_t *>(base) + data_offset);
  return data + slot_idx * SHM_MAX_POINTS * SHM_POINT_FLOATS;
}

// ─── SMPL shared memory ──────────────────────────────────────────────────────
//
//  [ SHMSMPLHeader ]
//  [ SHMSMPLSlot[SHM_SMPL_SLOTS] ]
//
static constexpr int SHM_SMPL_POSE_PARAMS = 72; // 24 joints × 3 axis-angle
static constexpr int SHM_SMPL_SHAPE_PARAMS = 10;
static constexpr int SHM_SMPL_TRANS_PARAMS = 3;

struct alignas(64) SHMSMPLSlot {
  std::atomic<uint32_t> state; // 0=free 1=writing 2=ready
  uint32_t valid;              // 1 if body was detected
  uint64_t timestamp_ns;
  uint32_t frame_id;
  uint8_t _pad[44];

  float pose[SHM_SMPL_POSE_PARAMS];   // 72 floats
  float betas[SHM_SMPL_SHAPE_PARAMS]; // 10 floats
  float trans[SHM_SMPL_TRANS_PARAMS]; // 3  floats
  float global_orient[3];             // root orientation (axis-angle)
};

struct alignas(64) SHMSMPLHeader {
  std::atomic<uint64_t> write_seq;
  uint32_t num_slots;
  uint8_t _pad[52];
};

inline constexpr size_t shm_smpl_total_size() {
  return sizeof(SHMSMPLHeader) + sizeof(SHMSMPLSlot) * SHM_SMPL_SLOTS;
}
inline SHMSMPLHeader *shm_smpl_header(void *base) {
  return reinterpret_cast<SHMSMPLHeader *>(base);
}
inline SHMSMPLSlot *shm_smpl_slot(void *base, int slot) {
  return reinterpret_cast<SHMSMPLSlot *>(static_cast<uint8_t *>(base) +
                                         sizeof(SHMSMPLHeader)) +
         slot;
}