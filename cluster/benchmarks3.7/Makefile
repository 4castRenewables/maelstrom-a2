# (JSC: SPLIT)
jube-split-juelich-test:
	jube run jube_training.yaml -t jwb jnvidia split test

jube-split-juelich-datamedium-test:
	jube run jube_training.yaml -t jwb jnvidia split test datamedium

jube-split-juelich-datalarge-test:
	jube run jube_training.yaml -t jwb jnvidia split test datalarge

## JSC: TRAINING
# JRC: H100
jube-train-juelich-h100-datamedium-test:
	jube run jube_training.yaml -t jwb h100 test datamedium

jube-train-juelich-h100-datamedium-test-double:
	jube run jube_training.yaml -t jwb h100 test datamedium gpudouble

jube-train-juelich-h100-datamedium-test-single:
	jube run jube_training.yaml -t jwb h100 test datamedium gpusingle

# JRC: Mi250
jube-train-juelich-mi250-datamedium:
	jube run jube_training.yaml -t jwb mi250 datamedium

jube-train-juelich-mi250-datamedium-quadro:
	jube run jube_training.yaml -t jwb mi250 datamedium gpuquadro

jube-train-juelich-mi250-datamedium-double:
	jube run jube_training.yaml -t jwb mi250 datamedium gpudouble

# JBooster: A100
jube-train-juelich-booster-test:
	jube run jube_training.yaml -t jwb jnvidia test

jube-train-juelich-booster-benchs-test:
	jube run jube_training.yaml -t jwb jnvidia test benchs

jube-train-juelich-booster-datamedium-test:
	jube run jube_training.yaml -t jwb jnvidia test datamedium

jube-train-juelich-booster-datamedium-test-double:
	jube run jube_training.yaml -t jwb jnvidia test datamedium gpudouble

jube-train-juelich-booster-datamedium-test-single:
	jube run jube_training.yaml -t jwb jnvidia test datamedium gpusingle

jube-train-juelich-booster-datamedium-benchs-test:
	jube run jube_training.yaml -t jwb jnvidia test datamedium benchs

jube-train-juelich-booster-datalarge-test:
	jube run jube_training.yaml -t jwb jnvidia test datalarge

# JRC: A100
jube-train-juelich-a100-datamedium-quadro:
	jube run jube_training.yaml -t a100 datamedium gpuquadro test
jube-train-juelich-a100-datamedium-double:
	jube run jube_training.yaml -t a100 datamedium gpudouble test
jube-train-juelich-a100-datamedium-single:
	jube run jube_training.yaml -t a100 datamedium gpusingle test

## (E4: SPLIT)
jube-split-e4-datamedium-test:
	jube run jube_training.yaml -t test split e4 datamedium gracehopper

## E4: TRAINING
# E4: Gracehopper
jube-train-e4-datamedium-test:
	jube run jube_training.yaml -t test e4power e4 datamedium gracehopper

# E4: A2
# Apptainer
jube-train-e4-gpua2-datamedium-apptainer-test-single:
	jube run jube_training.yaml -t test e4power e4 datamedium apptainer gpua2 gpuassignsingle gpusingle

jube-train-e4-gpua2-datamedium-apptainer-test-double:
	jube run jube_training.yaml -t test e4power e4 datamedium apptainer gpua2 gpudouble

# Conda
jube-train-e4-gpua2-datamedium-conda-test:
	jube run jube_training.yaml -t test e4power e4 datamedium conda gpua2

jube-train-e4-gpua2-datamedium-conda-test-gpudouble-single:
	jube run jube_training.yaml -t test e4power e4 datamedium conda gpua2 gpuassignsingle gpusingle

jube-train-e4-gpua2-datamedium-conda-test-gpudouble-double:
	jube run jube_training.yaml -t test e4power e4 datamedium conda gpua2 gpudouble

jube-train-e4-gpua2-datamedium-conda-test-gpudouble:
	jube run jube_training.yaml -t test e4power e4 datamedium conda gpua2 gpudouble
