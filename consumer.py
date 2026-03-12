"""
shm_consumer.py  —  legge point cloud e parametri SMPL dalla shared memory
                     scritta da main_shm.cpp

Layout SHM rispecchia shm_transport.h:
  /zed_pointclouds   → point cloud per camera
  /zed_smpl          → pose SMPL

Dipendenze: numpy, posix_ipc (pip install posix_ipc)
"""

import ctypes
import mmap
import os
import time
import posix_ipc
import numpy as np

# ─── tunabili (devono corrispondere a shm_transport.h) ───────────────────────
SHM_MAX_CAMS      = 4
SHM_PC_SLOTS      = 3
SHM_MAX_POINTS    = 500_000
SHM_POINT_FLOATS  = 9          # x y z r g b nx ny nz

SHM_SMPL_SLOTS    = 4
SHM_SMPL_POSE     = 72
SHM_SMPL_SHAPE    = 10
SHM_SMPL_TRANS    = 3

SHM_PC_NAME   = "/zed_pointclouds"
SHM_SMPL_NAME = "/zed_smpl"


# ─── ctypes structs (devono matchare le struct C++) ───────────────────────────

class SHMPCSlot(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("state",       ctypes.c_uint32),   # atomic, leggiamo come uint
        ("num_points",  ctypes.c_uint32),
        ("timestamp_ns",ctypes.c_uint64),
        ("cam_idx",     ctypes.c_uint32),
        ("frame_id",    ctypes.c_uint32),
        ("_pad",        ctypes.c_uint8 * 40),
    ]
assert ctypes.sizeof(SHMPCSlot) == 64, "SHMPCSlot size mismatch"

class SHMPCHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("write_seq",    ctypes.c_uint64 * SHM_MAX_CAMS),
        ("num_cams",     ctypes.c_uint32),
        ("max_points",   ctypes.c_uint32),
        ("point_floats", ctypes.c_uint32),
        ("num_slots",    ctypes.c_uint32),
        ("_pad",         ctypes.c_uint8 * 20),
    ]
assert ctypes.sizeof(SHMPCHeader) == 128, "SHMPCHeader size mismatch"

class SHMSMPLSlot(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("state",         ctypes.c_uint32),
        ("valid",         ctypes.c_uint32),
        ("timestamp_ns",  ctypes.c_uint64),
        ("frame_id",      ctypes.c_uint32),
        ("_pad",          ctypes.c_uint8 * 44),
        ("pose",          ctypes.c_float * SHM_SMPL_POSE),
        ("betas",         ctypes.c_float * SHM_SMPL_SHAPE),
        ("trans",         ctypes.c_float * SHM_SMPL_TRANS),
        ("global_orient", ctypes.c_float * 3),
    ]

class SHMSMPLHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("write_seq", ctypes.c_uint64),
        ("num_slots", ctypes.c_uint32),
        ("_pad",      ctypes.c_uint8 * 52),
    ]
assert ctypes.sizeof(SHMSMPLHeader) == 64, "SHMSMPLHeader size mismatch"


# ─── offset helpers (specchiamo le inline C++) ────────────────────────────────

def _pc_total_size():
    slots_size = ctypes.sizeof(SHMPCSlot) * SHM_MAX_CAMS * SHM_PC_SLOTS
    data_size  = (ctypes.sizeof(ctypes.c_float) * SHM_POINT_FLOATS
                  * SHM_MAX_POINTS * SHM_MAX_CAMS * SHM_PC_SLOTS)
    return ctypes.sizeof(SHMPCHeader) + slots_size + data_size

def _pc_slot_offset(cam: int, slot: int) -> int:
    return ctypes.sizeof(SHMPCHeader) + (cam * SHM_PC_SLOTS + slot) * ctypes.sizeof(SHMPCSlot)

def _pc_data_offset(cam: int, slot: int) -> int:
    slots_section = ctypes.sizeof(SHMPCSlot) * SHM_MAX_CAMS * SHM_PC_SLOTS
    idx = cam * SHM_PC_SLOTS + slot
    return (ctypes.sizeof(SHMPCHeader)
            + slots_section
            + idx * SHM_MAX_POINTS * SHM_POINT_FLOATS * 4)  # 4 = sizeof float32

def _smpl_total_size():
    return ctypes.sizeof(SHMSMPLHeader) + ctypes.sizeof(SHMSMPLSlot) * SHM_SMPL_SLOTS

def _smpl_slot_offset(slot: int) -> int:
    return ctypes.sizeof(SHMSMPLHeader) + slot * ctypes.sizeof(SHMSMPLSlot)


# ─── SHMReader ────────────────────────────────────────────────────────────────

class SHMReader:
    """
    Apre i segmenti SHM e fornisce metodi per leggere l'ultimo frame disponibile.
    Thread-safe per lettura (il C++ usa stati atomici).
    """

    def __init__(self):
        # Point cloud segment
        shm_pc = posix_ipc.SharedMemory(SHM_PC_NAME)
        self._mm_pc = mmap.mmap(shm_pc.fd, _pc_total_size(),
                                mmap.MAP_SHARED, mmap.PROT_READ)
        shm_pc.close_fd()

        self._pc_hdr = SHMPCHeader.from_buffer(self._mm_pc, 0)
        self._num_cams = self._pc_hdr.num_cams
        self._last_pc_seq = [0] * SHM_MAX_CAMS

        # SMPL segment
        shm_smpl = posix_ipc.SharedMemory(SHM_SMPL_NAME)
        self._mm_smpl = mmap.mmap(shm_smpl.fd, _smpl_total_size(),
                                  mmap.MAP_SHARED, mmap.PROT_READ)
        shm_smpl.close_fd()

        self._smpl_hdr  = SHMSMPLHeader.from_buffer(self._mm_smpl, 0)
        self._last_smpl_seq = 0

        print(f"[SHMReader] Opened {self._num_cams} camera(s)")

    def read_pointcloud(self, cam: int) -> dict | None:
        """
        Ritorna il punto cloud più recente per la camera `cam` come dict:
          {
            'points':   np.ndarray (N, 3) float32  xyz
            'colors':   np.ndarray (N, 3) float32  rgb
            'normals':  np.ndarray (N, 3) float32  nxyz
            'frame_id': int
            'timestamp_ns': int
          }
        Ritorna None se non c'è nessun frame nuovo da leggere.
        """
        hdr   = self._pc_hdr
        seq   = hdr.write_seq[cam]          # ultimo seq scritto
        if seq == self._last_pc_seq[cam]:
            return None                     # nessun frame nuovo

        slot = (seq - 1) % SHM_PC_SLOTS     # slot appena completato
        s_off = _pc_slot_offset(cam, slot)
        slot_obj = SHMPCSlot.from_buffer(self._mm_pc, s_off)

        # Aspetta che lo stato sia 2 (ready) con spin breve
        for _ in range(1000):
            if slot_obj.state == 2:
            	break
        else:
            return None   # timeout

        n          = slot_obj.num_points
        frame_id   = slot_obj.frame_id
        ts         = slot_obj.timestamp_ns
        d_off      = _pc_data_offset(cam, slot)

        # Zero-copy: numpy view sulla mmap
        raw = np.frombuffer(self._mm_pc,
                            dtype=np.float32,
                            count=n * SHM_POINT_FLOATS,
                            offset=d_off).reshape(n, SHM_POINT_FLOATS).copy()

        self._last_pc_seq[cam] = seq

        return {
            "points":        raw[:, 0:3],
            "colors":        raw[:, 3:6],
            "normals":       raw[:, 6:9],
            "frame_id":      frame_id,
            "timestamp_ns":  ts,
        }

    def read_smpl(self) -> dict | None:
        """
        Ritorna i parametri SMPL più recenti come dict:
          {
            'pose':          np.ndarray (72,) float32
            'betas':         np.ndarray (10,) float32
            'trans':         np.ndarray (3,)  float32
            'global_orient': np.ndarray (3,)  float32
            'valid':         bool
            'frame_id':      int
          }
        Ritorna None se nessun frame nuovo.
        """
        seq = self._smpl_hdr.write_seq
        if seq == self._last_smpl_seq:
            return None

        slot  = (seq - 1) % SHM_SMPL_SLOTS
        s_off = _smpl_slot_offset(slot)
        s     = SHMSMPLSlot.from_buffer(self._mm_smpl, s_off)

        for _ in range(1000):
            if s.state == 2:
                break
        else:
            return None

        self._last_smpl_seq = seq

        return {
            "pose":          np.array(s.pose,         dtype=np.float32),
            "betas":         np.array(s.betas,        dtype=np.float32),
            "trans":         np.array(s.trans,        dtype=np.float32),
            "global_orient": np.array(s.global_orient,dtype=np.float32),
            "valid":         bool(s.valid),
            "frame_id":      s.frame_id,
        }

    def close(self):
        self._mm_pc.close()
        self._mm_smpl.close()


# ─── esempio d'uso ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    reader = SHMReader()
    try:
        while True:
            for cam in range(reader._num_cams):
                pc = reader.read_pointcloud(cam)
                if pc:
                    print(f"[PC cam {cam}] frame={pc['frame_id']}  "
                          f"N={len(pc['points'])}  "
                          f"ts={pc['timestamp_ns']}")

            smpl = reader.read_smpl()
            if smpl and smpl["valid"]:
                print(f"[SMPL] frame={smpl['frame_id']}  "
                      f"trans={smpl['trans']}  "
                      f"betas={smpl['betas'][:3]}")

            time.sleep(0.001)   # polling a ~1kHz, oppure usa futex/eventfd
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()