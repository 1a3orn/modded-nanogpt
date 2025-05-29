base:
step:1770/1770 val_loss:3.2837 train_time:181057ms step_avg:102.29ms
step:1770/1770 val_loss:3.2814 train_time:181070ms step_avg:102.30ms


Notes:
- Hrm, not getting a lift even per step with just adding heron parameters
- My guess, is that this has to do with LR and embeds not working with muon?
- Trying with removing them from muon, changing LR