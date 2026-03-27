from __future__ import annotations

import math

import torch

from einops import einsum, reduce
from tensordict import TensorDict
from torch._tensor import Tensor

from rl4co.envs.scheduling.fjsp.env import FJSPEnv
from rl4co.utils.ops import gather_by_index

from .dataset import JSSPBucketedDataset
from .generator import JSSPFileGenerator, JSSPGenerator


class JSSPEnv(FJSPEnv):
    """Job-Shop Scheduling Problem (JSSP) environment
    At each step, the agent chooses a job. The operation to be processed next for the selected job is
    then executed on the associated machine. The reward is 0 unless the agent scheduled all operations of all jobs.
    In that case, the reward is (-)makespan of the schedule: maximizing the reward is equivalent to minimizing the makespan.
    NOTE: The JSSP is a special case of the FJSP, when the number of eligible machines per operation is equal to one for all
    operations. Therefore, this environment is a subclass of the FJSP environment.
    Observations:
        - time: current time
        - next_op: next operation per job
        - proc_times: processing time of operation-machine pairs
        - pad_mask: specifies padded operations
        - start_op_per_job: id of first operation per job
        - end_op_per_job: id of last operation per job
        - start_times: start time of operation (defaults to 0 if not scheduled)
        - finish_times: finish time of operation (defaults to INIT_FINISH if not scheduled)
        - job_ops_adj: adjacency matrix specifying job-operation affiliation
        - ops_job_map: same as above but using ids of jobs to indicate affiliation
        - ops_sequence_order: specifies the order in which operations have to be processed
        - ma_assignment: specifies which operation has been scheduled on which machine
        - busy_until: specifies until when the machine will be busy
        - num_eligible: number of machines that can process an operation
        - job_in_process: whether job is currently being processed
        - job_done: whether the job is done

    Constrains:
        the agent may not select:
        - jobs that are done already
        - jobs that are currently processed

    Finish condition:
        - the agent has scheduled all operations of all jobs

    Reward:
        - the negative makespan of the final schedule

    Args:
        generator: JSSPGenerator instance as the data generator
        generator_params: parameters for the generator
        mask_no_ops: if True, agent may not select waiting operation (unless instance is done)
    """

    name = "jssp"
    DEFAULT_SUPPORTED_SHAPES = ((10, 10), (15, 15), (20, 20))

    def __init__(
        self,
        generator: JSSPGenerator = None,
        generator_params: dict = {},
        mask_no_ops: bool = True,
        bucketed_training: bool = False,
        shape_buckets: list[list[int]] | None = None,
        **kwargs,
    ):
        self.bucketed_training = bucketed_training
        self.supported_shapes = tuple(
            tuple(shape)
            for shape in (shape_buckets if shape_buckets is not None else self.DEFAULT_SUPPORTED_SHAPES)
        )
        if len(self.supported_shapes) == 0:
            raise ValueError("shape_buckets must contain at least one shape")
        if len(set(self.supported_shapes)) != len(self.supported_shapes):
            raise ValueError(f"Duplicate shapes in shape_buckets: {self.supported_shapes}")

        self._generator_params = dict(generator_params)
        if generator is None:
            if generator_params.get("file_path", None) is not None:
                generator = JSSPFileGenerator(
                    supported_shapes=self.supported_shapes, **generator_params
                )
            else:
                generator = JSSPGenerator(**generator_params)
        if (generator.num_jobs, generator.num_mas) not in self.supported_shapes:
            raise ValueError(
                f"Unsupported JSSP shape {generator.num_jobs}x{generator.num_mas}. "
                f"Supported shapes: {self.supported_shapes}"
            )

        super().__init__(generator, generator_params, mask_no_ops, **kwargs)

    def _shape_counts(self, total_size: int) -> dict[tuple[int, int], int]:
        """Split dataset size approximately uniformly over configured shape buckets."""
        bucket_list = list(self.supported_shapes)
        n_buckets = len(bucket_list)
        base = total_size // n_buckets
        remainder = total_size % n_buckets
        counts = {}
        for idx, shape in enumerate(bucket_list):
            counts[shape] = base + (1 if idx < remainder else 0)
        return counts

    @staticmethod
    def _to_dataset_size(batch_size) -> int:
        if isinstance(batch_size, int):
            return int(batch_size)
        if isinstance(batch_size, (list, tuple)):
            if len(batch_size) == 0:
                return 0
            return int(math.prod(batch_size))
        raise TypeError(f"Unsupported batch_size type: {type(batch_size)}")

    def _build_bucketed_generated_dataset(self, size: int):
        if size <= 0:
            return JSSPBucketedDataset(samples=[], bucket_ids=[])

        samples: list[dict[str, torch.Tensor]] = []
        bucket_ids: list[str] = []
        counts = self._shape_counts(size)

        base_params = {
            k: v
            for k, v in self._generator_params.items()
            if k not in {"num_jobs", "num_machines", "file_path"}
        }
        for (num_jobs, num_machines), count in counts.items():
            if count <= 0:
                continue
            gen = JSSPGenerator(
                num_jobs=num_jobs,
                num_machines=num_machines,
                **base_params,
            )
            td = gen([count])
            bucket_id = f"{num_jobs}x{num_machines}"
            for i in range(count):
                sample = {k: v[i] for k, v in td.items()}
                sample["bucket_num_jobs"] = torch.tensor(num_jobs, dtype=torch.int64)
                sample["bucket_num_machines"] = torch.tensor(
                    num_machines, dtype=torch.int64
                )
                samples.append(sample)
                bucket_ids.append(bucket_id)

        return JSSPBucketedDataset(samples=samples, bucket_ids=bucket_ids)

    def dataset(self, batch_size=[], phase="train", filename=None):
        if not self.bucketed_training:
            return super().dataset(batch_size=batch_size, phase=phase, filename=filename)

        # Bucketed training currently supports generated data only.
        f = getattr(self, f"{phase}_file") if filename is None else filename
        if f is not None:
            raise NotImplementedError(
                "JSSP bucketed training currently supports generated data only (no file-based mixed-shape dataset)"
            )
        size = self._to_dataset_size(batch_size)
        return self._build_bucketed_generated_dataset(size)

    def _get_features(self, td):
        td = super()._get_features(td)
        # get the id of the machine that executes an operation:
        # (bs, ops, ma)
        ops_ma_adj = td["ops_ma_adj"].transpose(1, 2)
        # (bs, jobs, ma)
        ma_of_next_op = gather_by_index(ops_ma_adj, td["next_op"], dim=1)
        # (bs, jobs)
        td["next_ma"] = ma_of_next_op.argmax(-1)

        # adjacency matrix specifying neighbors of an operation, including its
        # predecessor and successor operations and operations on the same machine
        ops_on_same_ma_adj = einsum(
            td["ops_ma_adj"], td["ops_ma_adj"], "b m o1, b m o2 -> b o1 o2 "
        )
        # concat pred, succ and ops on same machine
        adj = torch.cat((td["ops_adj"], ops_on_same_ma_adj.unsqueeze(-1)), dim=-1).sum(-1)
        # mask padded operations and those scheduled
        mask = td["pad_mask"] + td["op_scheduled"]
        adj.masked_fill_(mask.unsqueeze(1), 0)
        td["adjacency"] = adj

        return td

    def get_action_mask(self, td: TensorDict) -> Tensor:
        action_mask = self._get_job_machine_availability(td)
        if self.mask_no_ops:
            # masking is only allowed if instance is finished
            no_op_mask = td["done"]
        else:
            # if no job is currently processed and instance is not finished yet, waiting is not allowed
            no_op_mask = (
                td["job_in_process"].any(1, keepdims=True) & (~td["done"])
            ) | td["done"]
        # reduce action mask to correspond with logit shape
        action_mask = reduce(action_mask, "bs j m -> bs j", reduction="all")
        # NOTE: 1 means feasible action, 0 means infeasible action
        # (bs, 1 + n_j)
        mask = torch.cat((no_op_mask, ~action_mask), dim=1)
        return mask

    def _translate_action(self, td):
        job = td["action"]
        op = gather_by_index(td["next_op"], job, dim=1)
        # get the machine that corresponds to the selected operation
        # td["ops_ma_adj"] shape: (bs, num_mas, n_ops_max)
        # op shape: (bs,)
        # For each batch element, we want: ops_ma_adj[batch_idx, :, op[batch_idx]]
        bs = td.size(0)
        batch_idx = torch.arange(bs, device=td.device)
        ma_adj = td["ops_ma_adj"][batch_idx, :, op]  # shape: (bs, num_mas)
        ma = ma_adj.argmax(dim=1)  # shape: (bs,)
        return job, op, ma

    @staticmethod
    def load_data(fpath, batch_size=[]):
        g = JSSPFileGenerator(fpath)
        return g(batch_size=batch_size)
