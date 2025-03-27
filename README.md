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

# Prove Phase - Performed by Users

```bash
cd ezkl/prove
```

Generate witness:

```bash
ezkl/prove $ ezkl gen-witness --data ../../new_data.json --compiled-circuit ../../iris_model.compiled  --output ../../witness.json --vk-path ../../vk.key --srs-path ../../kzg15.srs
```

This generates the file `witness.json` in the root folder of the project. The witness serves as a _trace_ of the computation, allowing the prover to demonstrate knowledge of all the steps involved in running the input through the model without revealing the specific values.

Then we generate the proof:

```bash
ezkl/prove $ ezkl prove --witness ../../witness.json --compiled-circuit ../../iris_model.compiled --pk-path ../../pk.key --proof-path ../../proof.json --srs-path ../../kzg15.srs
```

This will generate the file `proof.json` in the root folder of the project.

# Verify Phase - Performed by Verifier

```bash
ezkl/verify $ ezkl verify --settings-path ../setup/settings.json --proof-path ../../proof.json --vk-path ../../vk.key --srs-path ../../kzg15.srs
```

which prints something like this:

```bash
[*] [2025-03-27 08:36:10:965, ezkl::pfsys] - loaded verification key ✅
[*] [2025-03-27 08:36:10:972, ezkl::execute] - verify took 0.7
[*] [2025-03-27 08:36:10:972, ezkl::execute] - verified: true
[*] [2025-03-27 08:36:10:972, ezkl] - succeeded
```
