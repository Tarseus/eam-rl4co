include "globals.mzn";

int: num_jobs;
int: num_machines;
int: num_ops;

set of int: JOBS = 1..num_jobs;
set of int: MACHINES = 1..num_machines;
set of int: OPS = 1..num_ops;

array[JOBS, OPS] of MACHINES: machine_of;
array[JOBS, OPS] of int: duration;

int: horizon = sum(j in JOBS, o in OPS)(duration[j, o]);

array[JOBS, OPS] of var 0..horizon: start;
var 0..horizon: makespan;

constraint forall(j in JOBS, o in 1..num_ops - 1)(
    start[j, o] + duration[j, o] <= start[j, o + 1]
);

constraint forall(m in MACHINES)(
    disjunctive(
        [start[j, o] | j in JOBS, o in OPS where machine_of[j, o] == m],
        [duration[j, o] | j in JOBS, o in OPS where machine_of[j, o] == m]
    )
);

constraint forall(j in JOBS)(
    start[j, num_ops] + duration[j, num_ops] <= makespan
);

solve minimize makespan;

output [
    "makespan=", show(makespan), "\n"
];
