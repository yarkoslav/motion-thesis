# motion-thesis
Latent Motion Bachelor Thesis

Setup:
```bash
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
```
And get the sintel checkpoint for the FlowFormerPlusPlus (thus put it into the FlowFormerPlusPlus/checkpoints folder)

Commands to check Optical Flow Estimator (change path to the checkpoint in the FlowFormerPlusPlus/configs/submissions.py):
```bash
conda activate motion-thesis
cd FlowFormerPlusPlus
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
python visualize_flow.py --eval_type seq --seq_dir demo-frames --start_idx 16 --end_idx 25 --viz_root_dir viz_results
```

Commands to run train of flow VAE:
```bash
conda activate motion-thesis
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
python scripts/train_vae.py --base configs/autoencoder.yaml --train
python scripts/train_vae.py --base configs/autoencoder_flow.yaml --train
python scripts/train_vae.py --base configs/autoencoder_sanity_check.yaml --train
python scripts/train_vae.py --base configs/autoencoder_flow_sanity_check.yaml --train
```

Commands tu run eval of flow VAE:
```bash
conda activate motion-thesis
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
python scripts/test_vae.py --ckpt_path ckpt_path --config_path config_path --out_dir test_lwm/exp_name
```

Commands to run train of LWM:
```bash
conda activate motion-thesis
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
python scripts/train_lwm.py --config_path configs/lwm.yaml --train
python scripts/train_lwm.py --config_path configs/lwm_attn.yaml --train
python scripts/train_lwm.py --config_path configs/lwm_sanity_check.yaml --train
python scripts/train_lwm.py --config_path configs/lwm_attn_sanity_check.yaml --train
```

Commands to run eval of LWM:
```bash
conda activate motion-thesis
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
python scripts/test_lwm.py --ckpt_path ckpt_path --config_path config_path --out_dir test_lwm/exp_name
```

Data Download:
```bash
git clone https://github.com/m-bain/webvid.git
```
Also, you must have MPI downloaded
