#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import threading
import time
import uuid
import os
import torch
# better to add following code after import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    torch_npu = None
import logging
import torch.distributed as dist
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from logging.handlers import TimedRotatingFileHandler


class Diagnose:
    """
    Distributed large-scale EP is gradually becoming the main deployment strategy for MOE models. However, as the
        scale of EP increases, the risk of slowdowns in the Dispatch and Combine communication operators also rises.
    There are many factors, ranging from GPU hardware anomalies and imbalanced MOE computation to issues with communication
        links, all of which make the detection and localization of the DeepEP slow problems extremely challenging.
    To address this, we have designed a diagnosis module. Each rank collects the cumulative waiting time for receiving each token
        from other ranks and reports these statistics to rank 0. Based on the mean-normalized characteristics of the resulting
        analysis matrix, rank 0 can effectively detect and precisely localize slow anomalies in distributed communication.
    In addition, the impact of the overhead of this diagnostic module on performance can be ignored.

    Supports diagnosis of various slowdown scenarios, including:
        - 1. Slowdown caused by the destination rank.
        - 2. Slowdown caused by the source rank.
        - 3. Slowdown caused by the communication path between specific source and destination ranks.

    Maintains a statistical matrix of cumulative receive wait times: Matrix[src_rank, dst_rank], where each row represents a
        source rank and each column represents a destination rank.

    Example anomaly localization:
    1. Abnormal column 3: indicates destination rank 3 is slow.
    16   13   10  117   18   18   19   12
    10   19   11  118   16   16   16   13
    18   18   12  110   18   19   18   13
    13   18   16  112   12   11   18   18
    14   20   10  114   14   16   18   16
    20   20   15  114   19   13   15   18
    18   17   19  116   10   17   17   19
    15   17   20  118   13   13   15   14

    2. Abnormal row 6: indicates source rank 6 is slow.
    16   13   10   17   18   18   19   12
    10   19   11   18   16   16   16   13
    18   18   12   10   18   19   18   13
    13   18   16   12   12   11   18   18
    14   20   10   14   14   16   18   16
    20   20   15   14   19   13   15   18
    138  137  139  137  130  137  137  139
    15   17   20   18   13   13   15   14

    3. Abnormal entry (3, 4): indicates the path from src=3 to dst=4 is slow.
    16   13   10   17   18   18   19   12
    10   19   11   18   16   16   16   13
    18   18   12   10   18   19   18   15
    13   18   16   12   125  11   18   18
    14   20   10   14   14   16   18   16
    20   20   15   14   19   13   15   18
    18   17   19   17   10   17   17   19
    15   17   20   18   13   13   15   14

    Attributes:
        group: the communication group(i.e., the EP communication group).
        rank: the global rank number.
        group_size: the number of ranks in the group.
        interval: diagnose interval.
        enable_ll_diagnose: enable low latency mode diagnose.
        enable_normal_diagnose: enable normal mode diagnose.
        stop_diagnose: whether to stop diagnose.
        instance_id: diagnose instance id.
        logger: diagnose logger.
        ll_dispatch_wait_recv_cost_stats: a cumulative wait time for receiving each token.
                                                    shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
        ll_combine_wait_recv_cost_stats: a cumulative wait time for receiving each token.
                                                   shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
        normal_dispatch_wait_recv_cost_stats: a cumulative wait time for receiving each token.
                                                        shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
        normal_combine_wait_recv_cost_stats: a cumulative wait time for receiving each token.
                                                       shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
        gather_tensor: save all ranks diagnose stats to rank0.
        cpu_group: the communication of `gloo` group.

    Environment variables:
        DEEPEP_DIAGNOSE_ENABLE: determine diagnose enable switch from environment variable. Default 1.
        DEEPEP_DIAGNOSE_INTERVAL: controls the diagnose cycle period in seconds. Default 20.
        DEEPEP_DIAGNOSE_SYNC_STEP: controls the diagnose step counter. Default: 580.
        DEEPEP_DIAGNOSE_LOG_PATH: set the output file path for diagnose logs. Default ".".
        DEEPEP_DIAGNOSE_LOG_DETAILS: determine output the diagnose details info. Default "0".
        DEEPEP_DIAGNOSE_THRESHOLD_COL: determine threshold for abnormal columns. Default 3.0.
        DEEPEP_DIAGNOSE_THRESHOLD_ROW: determine threshold for abnormal rows. Default 3.0.
        DEEPEP_DIAGNOSE_THRESHOLD_POINT: determine threshold for abnormal individual points. Default 5.0.

    """

    def __init__(
            self,
            group: dist.ProcessGroup,
            interval: int = 20,
            enable_ll_diagnose: bool = True,
            enable_normal_diagnose: bool = False,
            enable_async: bool = False) -> None:
        """
        Initialize the diagnose.

        Arguments:
            group: the communication group(i.e., the EP communication group).
            interval: diagnose interval. Default 20.
            enable_ll_diagnose: enable low latency mode diagnose. Default `True`.
            enable_normal_diagnose: enable normal mode diagnose. Default `False`.
            enable_async: enable async diagnose mode. Default `False`.

        """

        # Check parameters
        assert group.size() != 0 and interval > 0, 'invalid parameter for diagnose'

        # Determine diagnose enable switch from environment variable
        enable_diagnose = os.getenv(
            "DEEPEP_DIAGNOSE_ENABLE", "1").lower() not in (
            "0", "false", "off")

        # Determine diagnose enable switch from environment variable
        self.enable_details = os.getenv(
            "DEEPEP_DIAGNOSE_LOG_DETAILS", "0").lower() not in (
            "0", "false", "off")

        # Determine threshold for abnormal columns
        self.thres_col = float(os.getenv("DEEPEP_DIAGNOSE_THRESHOLD_COL", 3.0))
        # Determine threshold for abnormal rows
        self.thres_row = float(os.getenv("DEEPEP_DIAGNOSE_THRESHOLD_ROW", 3.0))
        # Determine threshold for abnormal individual points
        self.thres_point = float(
            os.getenv(
                "DEEPEP_DIAGNOSE_THRESHOLD_POINT",
                5.0))

        # Initialize the diagnose
        self.group = group
        self.rank = group.rank()
        self.group_size = group.size()
        self.enable_ll_diagnose = enable_ll_diagnose and enable_diagnose
        self.enable_normal_diagnose = enable_normal_diagnose and enable_diagnose
        self.enable_async = enable_async
        # Controls the diagnose cycle period in seconds. Default: 20
        self.interval = int(os.getenv("DEEPEP_DIAGNOSE_INTERVAL", interval))
        # Controls the diagnose step counter. Default: 580
        self.sync_step = np.uint64(os.getenv("DEEPEP_DIAGNOSE_SYNC_STEP", 580))
        self.stop_diagnose = threading.Event()

        self.logger = Diagnose._setup_logger_internal(rank=self.rank)
        # TODO: Use pinned memory optimization
        if self.enable_ll_diagnose:
            self.ll_dispatch_wait_recv_cost_stats = torch.zeros(
                (self.group_size, ), dtype=torch.int64, device='cuda')
            self.ll_combine_wait_recv_cost_stats = torch.zeros(
                (self.group_size, ), dtype=torch.int64, device='cuda')
            self.sync_ll_step_counter = np.uint64(0)
        if self.enable_normal_diagnose:
            self.normal_dispatch_wait_recv_cost_stats = torch.zeros(
                (self.group_size, ), dtype=torch.int64, device='cuda')
            self.normal_combine_wait_recv_cost_stats = torch.zeros(
                (self.group_size, ), dtype=torch.int64, device='cuda')
            self.sync_normal_step_counter = np.uint64(0)
        if self.enable_ll_diagnose or self.enable_normal_diagnose:
            if self.rank == 0:
                ubytes = torch.tensor(
                    list(
                        uuid.uuid4().bytes),
                    dtype=torch.uint8,
                    device='cuda')
            else:
                ubytes = torch.empty(16, dtype=torch.uint8, device='cuda')
            # Initialize the instance id
            dist.broadcast(ubytes, src=0, group=group)
            self.instance_id = uuid.UUID(bytes=ubytes.cpu().numpy().tobytes())
            # Initialize the stats tensor
            stats_list = [
                self.ll_dispatch_wait_recv_cost_stats,
                self.ll_combine_wait_recv_cost_stats]
            # Using gloo to avoid affecting GPU communication when enable_async
            # mode
            self.cpu_group = dist.new_group(ranks=list(
                range(self.group_size)), backend='gloo')
            stack_tensor = torch.stack(stats_list, dim=0)
            target_device = "cpu" if enable_async else stack_tensor.device
            self.gather_tensor = [
                torch.zeros_like(
                    stack_tensor, device=target_device) for _ in range(
                    self.group_size)] if self.rank == 0 else None

    def get_stats_ll_stats_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the cumulative wait time for receiving each token under low-latency mode for statistical purposes,
        which is useful for detecting and precisely localizing slow anomalies.
        The shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.

        Returns:
            tuple[0]: ll_dispatch_wait_recv_cost_stats.
            tuple[1]: ll_combine_wait_recv_cost_stats.
        """
        return (
            self.ll_dispatch_wait_recv_cost_stats if self.enable_ll_diagnose else None,
            self.ll_combine_wait_recv_cost_stats if self.enable_ll_diagnose else None)

    def get_stats_normal_stats_tensor(
            self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the cumulative wait time for receiving each token under normal mode for statistical purposes,
        which is useful for detecting and precisely localizing slow anomalies.
        The shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.

        Returns:
            tuple[0]: normal_dispatch_wait_recv_cost_stats.
            tuple[1]: normal_combine_wait_recv_cost_stats.
        """
        return (
            self.normal_dispatch_wait_recv_cost_stats if self.enable_normal_diagnose else None,
            self.normal_combine_wait_recv_cost_stats if self.enable_normal_diagnose else None)

    def get_all_stats_tensor(self):
        """
        Get the all ranks stats tensor.

        Returns:
            gather_tensor: the gather_tensor for the instance.
        """
        if not self.enable_ll_diagnose and not self.enable_normal_diagnose:
            return None
        return torch.stack(self.gather_tensor,
                           dim=0).numpy() if self.rank == 0 else None

    def get_instance_id(self):
        """
        Get the instance id.

        Returns:
            instance_id: diagnose instance id.
        """
        return self.instance_id

    @staticmethod
    def _setup_logger_internal(
            log_prefix="diagnose",
            when="midnight",
            interval=1,
            backupCount=2,
            rank=None):
        logger = logging.getLogger(
            f'diagnose_logger{"" if rank is None else f"_rank{rank}"}')
        # stops searching up the hierarchy whenever a logger with the
        # ‘propagate’ attribute set to false is found.
        logger.propagate = False
        logger.setLevel(logging.INFO)
        log_name = f"{log_prefix}{'' if rank is None else f'_rank{rank}'}.log"
        # Set the output file path for diagnose logs. Default ".".
        log_dir = os.environ.get('DEEPEP_DIAGNOSE_LOG_PATH', '.')
        os.makedirs(log_dir, exist_ok=True)
        file = os.path.join(log_dir, log_name)
        handler = TimedRotatingFileHandler(
            file,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding='utf-8')
        formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S')
        handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(handler)
        return logger

    @staticmethod
    def diagnose_matrix(
        mat, thres_col=3.0, thres_row=3.0, thres_point=5.0,
        suppress_points_in_strong_rowscols=True
    ):
        """
        Detect abnormal columns, rows, and individual points in a 2D wait-time matrix.

        Arguments:
            mat (np.ndarray): 2D array where mat[i, j] is the waiting time of source i for destination j.
            thres_col (float): Threshold for abnormal columns.
            thres_row (float): Threshold for abnormal rows.
            thres_point (float): Threshold for abnormal individual points.
            suppress_points_in_strong_rowscols (bool): If True, exclude points already in detected abnormal rows/columns.

        Returns:
            dict: {
                "abnormal_cols": List[List[int, float, float]],  # abnormal column indices, [col_index, mean_value, normalized_value]
                "abnormal_rows": List[List[int, float, float]],  # abnormal row indices, [rol_index, mean_value, normalized_value]
                "abnormal_points": List[List[int, int, float, float]]  # abnormal points, [row, col, value, normalized_value]
            }
        """
        # 1. Check for abnormal columns
        col_means = mat.mean(axis=0)
        # z_col = (col_means - col_means.mean()) / (col_means.std() + 1e-8)
        z_col = col_means / (col_means.mean() + 1e-8)
        abnormal_cols = [
            [j, col_means[j], z_col[j]]
            for j in np.where(z_col > thres_col)[0]
        ]

        # 2. Check for abnormal rows
        row_means = mat.mean(axis=1)
        # z_row = (row_means - row_means.mean()) / (row_means.std() + 1e-8)
        z_row = row_means / (row_means.mean() + 1e-8)
        abnormal_rows = [
            [i, row_means[i], z_row[i]]
            for i in np.where(z_row > thres_row)[0]
        ]

        # 3. Check for abnormal single points
        # z_all = (mat - mat.mean()) / (mat.std() + 1e-8)
        z_all = mat / (mat.mean() + 1e-8)
        # Get all positions with z-score > threshold
        abnormal_points = [
            [i, j, mat[i, j], z_all[i, j]]
            for i in range(mat.shape[0])
            for j in range(mat.shape[1])
            if z_all[i, j] > thres_point
        ]
        # Optionally remove points that are in already detected abnormal rows
        # or columns
        if suppress_points_in_strong_rowscols:
            strong_rows = [row[0] for row in abnormal_rows]
            strong_cols = [col[0] for col in abnormal_cols]
            abnormal_points = [
                [i, j, v, z] for [i, j, v, z] in abnormal_points
                if i not in strong_rows and j not in strong_cols
            ]
        # 4. Return for automatic processing
        return {
            "abnormal_cols": abnormal_cols,
            "abnormal_rows": abnormal_rows,
            "abnormal_points": abnormal_points
        }

    def _reset_ll_stats(self):
        # Make LL dispatch/combine stats tensor zero
        self.ll_dispatch_wait_recv_cost_stats.zero_()
        self.ll_combine_wait_recv_cost_stats.zero_()

    def _reset_normal_stats(self):
        # Make normal dispatch/combine stats tensor zero
        self.normal_dispatch_wait_recv_cost_stats.zero_()
        self.normal_combine_wait_recv_cost_stats.zero_()

    def _diagnose_internal(self):
        while not self.stop_diagnose.is_set():
            time.sleep(self.interval)
            try:
                if self.enable_ll_diagnose:
                    self._gather_diagnose_stats_internal(
                        [self.ll_dispatch_wait_recv_cost_stats, self.ll_combine_wait_recv_cost_stats])
                    # Make LL dispatch/combine stats tensor zero
                    self._reset_ll_stats()

                if self.enable_normal_diagnose:
                    self._gather_diagnose_stats_internal(
                        [
                            self.normal_dispatch_wait_recv_cost_stats,
                            self.normal_combine_wait_recv_cost_stats])
                    # Make normal dispatch/combine stats tensor zero
                    self._reset_normal_stats()
            except Exception as e:
                self.logger.info(
                    f"[Diagnose] InstanceID: {self.instance_id} EPSize: {self.group_size} Rank: {self.rank} deepep/dist error: {e}, diagnose thread exit.")
                logging.shutdown()
                break

    def _gather_diagnose_stats_internal(
            self, stats_list) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if torch.cuda.is_current_stream_capturing():
            return results
        if not self.enable_async:
            group = self.group
            stats_tensor = torch.stack(stats_list, dim=0)   # (N, num_ranks)
        else:
            group = self.cpu_group
            stats_tensor = torch.stack(
                stats_list, dim=0).cpu()   # (N, num_ranks)
        dist.gather(
            stats_tensor,
            gather_list=self.gather_tensor,
            group=group,
            dst=0)
        # The numpy is not permitted when stream is capturing
        if self.rank == 0 and (not torch.cuda.is_current_stream_capturing()):
            if self.gather_tensor[0].is_cuda:
                stats_arr = torch.stack(
                    [it.cpu() for it in self.gather_tensor], dim=0).numpy()
            else:
                stats_arr = torch.stack(self.gather_tensor, dim=0).numpy()
            for i, name in enumerate(["Dispatch", "Combine"]):
                res = Diagnose.diagnose_matrix(
                    stats_arr[:, i, :], thres_col=self.thres_col, thres_row=self.thres_row, thres_point=self.thres_point)
                results.append(res)
                self.logger.info(
                    f"[Diagnose] InstanceID: {self.instance_id} EPSize: {self.group_size}, diagnose: {res}, {name} Wait Recv Cost Per Token Matrix[src_rank, dst_rank]")
                if self.enable_details:
                    for idx, row in enumerate(stats_arr[:, i, :]):
                        self.logger.info(
                            f"rank={idx} [{' '.join(f'{val:8d}' for val in row)}]")
        return results

    def diagnose_ll_sync(self, diagnose_step: int = 0) -> List[Dict[str, Any]]:
        """
        Perform synchronous diagnosis for low latency (LL) DeepEP mode every `step_round` steps.

        This method runs synchronously and returns diagnosis results for
        both dispatch and combine phases.
        All diagnosis results are gathered to rank 0.

        Notes:
        In synchronous (sync) mode, **all ranks in the EP domain must call this
        function at the same code location (for example, once every 580 steps)**.
        Failing to do so can result in deadlocks or hangs due to distributed synchronization.

        Returns:
            List[Dict[str, Any]]: A list containing diagnosis information for dispatch and combine.
                                  Each element corresponds to one phase and includes
                                  abnormal columns, rows, and points.
        """
        assert self.enable_async is False, "diagnose_ll_sync() can only be called when 'self.enable_async' is False."
        if not self.enable_ll_diagnose:
            return None

        # The value priority set by diagnose_step is higher than self.sync_step
        step_round = diagnose_step if diagnose_step != 0 else self.sync_step
        self.sync_ll_step_counter += 1
        if self.sync_ll_step_counter % step_round != 0:
            return None
        res = self._gather_diagnose_stats_internal(
            [self.ll_dispatch_wait_recv_cost_stats, self.ll_combine_wait_recv_cost_stats])

        # Make LL dispatch/combine stats tensor zero
        self._reset_ll_stats()
        return res

    def diagnose_normal_sync(
            self, diagnose_step: int = 0) -> List[Dict[str, Any]]:
        """
        Perform synchronous diagnosis for normal DeepEP mode every `step_round` steps.

        This method runs synchronously and returns diagnosis results for
        both dispatch and combine phases.
        All diagnosis results are gathered to rank 0.

        Notes:
        In synchronous (sync) mode, **all ranks in the EP domain must call this
        function at the same code location (for example, once every 580 steps)**.
        Failing to do so can result in deadlocks or hangs due to distributed synchronization.

        Returns:
            List[Dict[str, Any]]: A list containing diagnosis information for dispatch and combine.
                                  Each element corresponds to one phase and includes
                                  abnormal columns, rows, and points.
        """
        assert self.enable_async is False, "diagnose_normal_sync() can only be called when 'self.enable_async' is False."
        if not self.enable_normal_diagnose:
            return None

        # The value priority set by diagnose_step is higher than self.sync_step
        step_round = diagnose_step if diagnose_step != 0 else self.sync_step
        self.sync_normal_step_counter += 1
        if self.sync_normal_step_counter % step_round != 0:
            return None

        res = self._gather_diagnose_stats_internal(
            [
                self.normal_dispatch_wait_recv_cost_stats,
                self.normal_combine_wait_recv_cost_stats])

        # Make normal dispatch/combine stats tensor zero
        self._reset_normal_stats()
        return res

    def start_async_diagnose(self):
        """
        Start the asynchronous diagnosis thread for both low latency and normal DeepEP modes.
        This method launches a background thread which will periodically perform diagnosis.
        All diagnosis results are gathered to rank 0.

        Returns:
            thread: The started diagnosis thread object.

        """
        assert self.enable_async is True, "start_async_diagnose() can only be called when 'self.enable_async' is True."

        if not self.enable_ll_diagnose and not self.enable_normal_diagnose:
            return None

        t = threading.Thread(target=self._diagnose_internal, daemon=True)
        t.start()
        return t

    def stop_async_diagnose(self):
        """
        Stop the asynchronous diagnosis.

        This method signals the background diagnosis thread (if running) to stop.
        """
        assert self.enable_async is True, "stop_async_diagnose() can only be called when 'self.enable_async' is True."

        if not self.enable_ll_diagnose and not self.enable_normal_diagnose:
            return None

        self.stop_diagnose.set()
