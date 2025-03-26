# Setup Phase - Performed by Developers

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

Then we compile the model:

```bash
ezkl/setup $ ezkl compile-circuit --model ../../iris_model.onnx --compiled-circuit ../../iris_model.compiled
```

This will generate the file `iris_model.compiled` in the root folder of the project.

Next we download the SRS file:

```bash
ezkl/setup $ ezkl get-srs --srs-path ../../kzg15.srs
```

This uses the file `settings.json` and generates the file `../../kzg15.srs`.

Final step in the **setup** phase is to run the `ezkl setup`.

```bash
ezkl/setup $ ezkl setup --compiled-circuit ../../iris_model.compiled --srs-path ../../kzg15.srs --vk-path ../../vk.key --pk-path ../../pk.key
```

The `pk.key` is used for proving and the the `vk.key` is used for
verifying.
