CAB-F train/infer module root hosted inside `img_tools`.

Purpose:
- Replace the previous external `D:\project\tianwei\train_model\modules` dependency
- Provide the in-repo home for `sew_point` and `sew_point_conntect`
- Keep CAB-F training / inference code and GUI workflow in one repository

Current state:
- The workflow config and initialization now point to this directory by default
- Full migration of the original training / inference script bodies is still pending

Expected future packages:
- `sew_point`
- `sew_point_conntect`
