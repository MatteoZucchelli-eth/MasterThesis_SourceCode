# BSD 3-Clause License
#
# Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import ctypes
import os
import platform
import struct

from ament_index_python.packages import get_package_share_directory
from unitree_hg.msg import LowCmd, LowState


# NOTE(@naefjo): Implement CRC for the hand messages. Is this needed?
class CRC:
    """Source: https://github.com/unitreerobotics/unitree_sdk2_python/blob/6cf6a184713ab0fc69336f5106f9578a620a677e/unitree_sdk2py/utils/crc.py"""

    def __init__(self):
        # 4 bytes aligned, little-endian format.
        # size 812
        self.__packFmtLowCmd = "<4B4IH2x" + "B3x5f3I" * 20 + "4B" + "55Bx2I"
        # size 1180
        self.__packFmtLowState = (
            "<4B4IH2x"
            + "13fb3x"
            + "B3x7fb3x3I" * 20
            + "4BiH4b15H"
            + "8hI41B3xf2b2x2f4h2I"
        )
        # size 1004
        self.__packFmtHGLowCmd = "<2B2x" + "B3x5fI" * 35 + "5I"
        # size 2092
        self.__packFmtHGLowState = "<2I2B2xI" + "13fh2x" + "B3x4f2hf7I" * 35 + "40B5I"

        self.platform = platform.system()
        lib_dir = os.path.join(get_package_share_directory("g1_common"), "lib")
        if self.platform == "Linux":
            if platform.machine() == "x86_64":
                self.crc_lib = ctypes.CDLL(os.path.join(lib_dir, "crc_amd64.so"))
            elif platform.machine() == "aarch64":
                self.crc_lib = ctypes.CDLL(os.path.join(lib_dir, "crc_aarch64.so"))

            self.crc_lib.crc32_core.argtypes = (
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint32,
            )
            self.crc_lib.crc32_core.restype = ctypes.c_uint32

    def Crc(self, msg):
        if isinstance(msg, LowCmd):
            return self.__Crc32(self.__PackHGLowCmd(msg))
        elif isinstance(msg, LowState):
            return self.__Crc32(self.__PackHGLowState(msg))
        else:
            raise TypeError("unknown IDL message type to crc")

    def __PackHGLowCmd(self, cmd: LowCmd):
        origData = []
        origData.append(cmd.mode_pr)
        origData.append(cmd.mode_machine)

        for motor_cmd in cmd.motor_cmd:
            origData.append(motor_cmd.mode)
            origData.append(motor_cmd.q)
            origData.append(motor_cmd.dq)
            origData.append(motor_cmd.tau)
            origData.append(motor_cmd.kp)
            origData.append(motor_cmd.kd)
            origData.append(motor_cmd.reserve)

        origData.extend(cmd.reserve)
        origData.append(cmd.crc)

        return self.__Trans(struct.pack(self.__packFmtHGLowCmd, *origData))

    def __PackHGLowState(self, state: LowState):
        origData = []
        origData.extend(state.version)
        origData.append(state.mode_pr)
        origData.append(state.mode_machine)
        origData.append(state.tick)

        origData.extend(state.imu_state.quaternion)
        origData.extend(state.imu_state.gyroscope)
        origData.extend(state.imu_state.accelerometer)
        origData.extend(state.imu_state.rpy)
        origData.append(state.imu_state.temperature)

        for motor_state in state.motor_state:
            origData.append(motor_state.mode)
            origData.append(motor_state.q)
            origData.append(motor_state.dq)
            origData.append(motor_state.ddq)
            origData.append(motor_state.tau_est)
            origData.extend(motor_state.temperature)
            origData.append(motor_state.vol)
            origData.extend(motor_state.sensor)
            origData.append(motor_state.motorstate)
            origData.extend(motor_state.reserve)

        origData.extend(state.wireless_remote)
        origData.extend(state.reserve)
        origData.append(state.crc)
        return self.__Trans(struct.pack(self.__packFmtHGLowState, *origData))

    def __Trans(self, packData):
        calcData = []
        calcLen = (len(packData) >> 2) - 1

        for i in range(calcLen):
            d = (
                (packData[i * 4 + 3] << 24)
                | (packData[i * 4 + 2] << 16)
                | (packData[i * 4 + 1] << 8)
                | (packData[i * 4])
            )
            calcData.append(d)

        return calcData

    def _crc_py(self, data):
        bit = 0
        crc = 0xFFFFFFFF
        polynomial = 0x04C11DB7

        for i in range(len(data)):
            bit = 1 << 31
            current = data[i]

            for b in range(32):
                if crc & 0x80000000:
                    crc = (crc << 1) & 0xFFFFFFFF
                    crc ^= polynomial
                else:
                    crc = (crc << 1) & 0xFFFFFFFF

                if current & bit:
                    crc ^= polynomial

                bit >>= 1

        return crc

    def _crc_ctypes(self, data):
        uint32_array = (ctypes.c_uint32 * len(data))(*data)
        length = len(data)
        crc = self.crc_lib.crc32_core(uint32_array, length)
        return crc

    def __Crc32(self, data):
        if self.platform == "Linux":
            return self._crc_ctypes(data)
        else:
            return self._crc_py(data)
