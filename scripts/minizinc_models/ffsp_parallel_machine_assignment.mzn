int: num_jobs;
int: num_stages;
int: num_machines;

set of int: JOBS = 1..num_jobs;
set of int: STAGES = 1..num_stages;
set of int: MACHINES = 1..num_machines;

array[JOBS, STAGES, MACHINES] of int: duration;

int: horizon = sum(j in JOBS, s in STAGES, m in MACHINES)(duration[j, s, m]);

array[JOBS, STAGES] of var 0..horizon: start;
array[JOBS, STAGES] of var 0..horizon: finish;
array[JOBS, STAGES, MACHINES] of var bool: assign;
array[JOBS, JOBS, STAGES, MACHINES] of var bool: before;
var 0..horizon: makespan;

constraint forall(j in JOBS, s in STAGES)(
    sum(m in MACHINES)(bool2int(assign[j, s, m])) = 1
);

constraint forall(j in JOBS, s in STAGES)(
    finish[j, s] = start[j, s] + sum(m in MACHINES)(duration[j, s, m] * bool2int(assign[j, s, m]))
);

constraint forall(j in JOBS, s in 1..num_stages - 1)(
    finish[j, s] <= start[j, s + 1]
);

constraint forall(s in STAGES, m in MACHINES, j1 in JOBS, j2 in JOBS where j1 < j2)(
    start[j1, s] + duration[j1, s, m] <= start[j2, s] +
    horizon * (3 - bool2int(assign[j1, s, m]) - bool2int(assign[j2, s, m]) - bool2int(before[j1, j2, s, m]))
);

constraint forall(s in STAGES, m in MACHINES, j1 in JOBS, j2 in JOBS where j1 < j2)(
    start[j2, s] + duration[j2, s, m] <= start[j1, s] +
    horizon * (2 - bool2int(assign[j1, s, m]) - bool2int(assign[j2, s, m]) + bool2int(before[j1, j2, s, m]))
);

constraint forall(j in JOBS)(
    finish[j, num_stages] <= makespan
);

solve minimize makespan;

output [
    "makespan=", show(makespan), "\n"
];
