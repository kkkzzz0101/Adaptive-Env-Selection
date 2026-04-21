# 512/64 Validate Results

- run: `qwen05b_random_mixed_formal_seed42`
- config: `random + grpo`, `max_prompt_length=512`, `max_response_length=64`, `steps=400`

## val/test_score
- `val/test_score/countdown_diff_1`: `0.000`
- `val/test_score/countdown_diff_2`: `0.000`
- `val/test_score/countdown_diff_3`: `0.000`
- `val/test_score/countdown_diff_4`: `0.000`
- `val/test_score/kk_logic_3`: `-3.000`
- `val/test_score/kk_logic_4`: `-3.000`
- `val/test_score/kk_logic_5`: `-3.000`
- `val/test_score/kk_logic_6`: `-3.000`
- `val/test_score/kk_logic_7`: `-3.000`
- `val/test_score/kk_logic_8`: `-3.000`
- `val/test_score/math_train_level_1`: `0.000`
- `val/test_score/math_train_level_2`: `0.000`
- `val/test_score/math_train_level_3`: `0.000`
- `val/test_score/math_train_level_4`: `0.000`
- `val/test_score/math_train_level_5`: `0.000`
- `val/test_score/zebra_houses_3`: `0.267`
- `val/test_score/zebra_houses_4`: `0.300`
- `val/test_score/zebra_houses_5`: `0.118`

## Raw Step 400 Line
```
step:400 - val/test_score/zebra_houses_3:0.267 - val/test_score/math_train_level_5:0.000 - val/test_score/kk_logic_6:-3.000 - val/test_score/kk_logic_5:-3.000 - val/test_score/countdown_diff_3:0.000 - val/test_score/zebra_houses_5:0.118 - val/test_score/math_train_level_3:0.000 - val/test_score/countdown_diff_2:0.000 - val/test_score/zebra_houses_4:0.300 - val/test_score/kk_logic_8:-3.000 - val/test_score/math_train_level_2:0.000 - val/test_score/math_train_level_4:0.000 - val/test_score/kk_logic_3:-3.000 - val/test_score/kk_logic_4:-3.000 - val/test_score/math_train_level_1:0.000 - val/test_score/countdown_diff_4:0.000 - val/test_score/kk_logic_7:-3.000 - val/test_score/countdown_diff_1:0.000 - val/test_score_extra/zebra_houses_3/format_score:0.000 - val/test_score_extra/math_train_level_5/format_score:0.000 - val/test_score_extra/kk_logic_6/format_score:-1.000 - val/test_score_extra/kk_logic_5/format_score:-1.000 - val/test_score_extra/countdown_diff_3/format_score:0.000 - val/test_score_extra/zebra_houses_5/format_score:0.000 - val/test_score_extra/math_train_level_3/format_score:0.000 - val/test_score_extra/countdown_diff_2/format_score:0.000 - val/test_score_extra/zebra_houses_4/format_score:0.000 - val/test_score_extra/kk_logic_8/format_score:-1.000 - val/test_score_extra/math_train_level_2/format_score:0.000 - val/test_score_extra/math_train_level_4/format_score:0.000 - val/test_score_extra/kk_logic_3/format_score:-1.000 - val/test_score_extra/kk_logic_4/format_score:-1.000 - val/test_score_extra/math_train_level_1/format_score:0.000 - val/test_score_extra/countdown_diff_4/format_score:0.000 - val/test_score_extra/kk_logic_7/format_score:-1.000 - val/test_score_extra/countdown_diff_1/format_score:0.000 - val/test_score_extra/zebra_houses_3/answer_score:0.000 - val/test_score_extra/math_train_level_5/answer_score:0.000 - val/test_score_extra/kk_logic_6/answer_score:-2.000 - val/test_score_extra/kk_logic_5/answer_score:-2.000 - val/test_score_extra/countdown_diff_3/answer_score:0.000 - val/test_score_extra/zebra_houses_5/answer_score:0.000 - val/test_score_extra/math_train_level_3/answer_score:0.000 - val/test_score_extra/countdown_diff_2/answer_score:0.000 - val/test_score_extra/zebra_houses_4/answer_score:0.000 - val/test_score_extra/kk_logic_8/answer_score:-2.000 - val/test_score_extra/math_train_level_2/answer_score:0.000 - val/test_score_extra/math_train_level_4/answer_score:0.000 - val/test_score_extra/kk_logic_3/answer_score:-2.000 - val/test_score_extra/kk_logic_4/answer_score:-2.000 - val/test_score_extra/math_train_level_1/answer_score:0.000 - val/test_score_extra/countdown_diff_4/answer_score:0.000 - val/test_score_extra/kk_logic_7/answer_score:-2.000 - val/test_score_extra/countdown_diff_1/answer_score:0.000
```