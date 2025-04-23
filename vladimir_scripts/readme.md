# temp just in case
<!-- git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout ae122b1cbde96c871fb74611363e04eecfbcce03
conda create -n vllm_int_0.7.4_test_2_testbrench python=3.10 -y
conda activate vllm_int_0.7.4_test_2_testbrench # conda remove -n vllm_int_0.7.4_test_2 --all -y 
VLLM_USE_PRECOMPILED=1 pip install --editable .
pip install librosa
pip install ipykernel
python -m ipykernel install --user --name vllm_int_0.7.4_test_2
unset LD_LIBRARY_PATH

export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/ae122b1cbde96c871fb74611363e04eecfbcce03/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable . -->


# install vllm
```
git clone https://github.com/vladimiralbrekhtccr/vllm_0.7.4_audio_integration.git
cd vllm_0.7.4_audio_integration
git checkout ae122b1c-branch
conda create -n vllm_int_0.7.4 python=3.10 -y
conda activate vllm_int_0.7.4 
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/ae122b1cbde96c871fb74611363e04eecfbcce03/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .
pip install librosa
pip install ipykernel
```

# run vllm server
```
./vllm_server_avlm.sh
```


# run inference
```
python send_req_audio.py
```