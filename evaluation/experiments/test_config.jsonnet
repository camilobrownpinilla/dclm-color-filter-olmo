/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

local rc_20tasks = import 'task_sets/test_sets/test_rc20_tasks.libsonnet';
local gen_tasks = import 'task_sets/test_sets/test_gen_tasks.libsonnet';
local ppl_suite = import 'task_sets/test_sets/test_eval_suite_ppl_val_v2_small.libsonnet';

local gsheet = "auto-gsheet-test";

// Models to evaluate

local models = [
    {
        model_path: "test_fixtures/test-olmo-model", //"s3://ai2-llm/test_fixtures/olmo-1b"
        gpus_needed: 1
    },
    {
        model_path: "sshleifer/tiny-gpt2",
        gpus_needed: 1
    }
];

local task_sets = [
    rc20_tasks.task_set,
    gen_tasks.task_set,
    ppl_suite.task_set
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}
