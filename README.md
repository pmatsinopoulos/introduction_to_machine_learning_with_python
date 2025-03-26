```bash
$ cd ezkl/setup

ezkl/setup $ ezkl gen-settings --model ../../iris_model.onnx
```

This process generates the file [./ezkl/setup/settings.json](./ezkl/setup/settings.json)

Then we have to calibrate the settings. This process takes as input the
`settings.json` file and writes to `calibrated_settings.json`.

```bash

ezkl/setup $ ezkl calibrate-settings --data ../../calibration_data.json --model ../../iris_model.onnx  -O settings.json
```
This updates the file `settings.json`.
