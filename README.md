# FantasyID: Face Knowledge Enhanced ID-Preserving Video Generation

[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://fantasy-amap.github.io/fantasy-id/) 
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2502.13995)
[![arXiv](https://img.shields.io/badge/Arxiv-2502.13995-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2502.13995) 

## Abstract

> Tuning-free approaches adapting large-scale pre-trained video diffusion models for identity-preserving text-to-video generation (IPT2V) have gained popularity recently due to their efficacy and scalability. However, significant challenges remain to achieve satisfied facial dynamics while keeping the identity unchanged. In this work, we present a novel tuning-free IPT2V framework by enhancing face knowledge of the pre-trained video model built on diffusion transformers (DiT), dubbed FantasyID. Essentially, 3D facial geometry prior is incorporated to ensure plausible facial structures during video synthesis. To prevent the model from learning ``copy-paste'' shortcuts that simply replicate reference face across frames, a multi-view face augmentation strategy is devised to capture diverse 2D facial appearance features, hence increasing the dynamics over the facial expressions and head poses. Additionally, after blending the 2D and 3D features as guidance, instead of naively employing adapter to inject guidance cues into DiT layers, a learnable layer-aware adaptive mechanism is employed to selectively inject the fused features into each individual DiT layers, facilitating balanced modeling of identity preservation and motion dynamics. Experimental results validate our model’s superiority over the current tuning-free IPT2V methods.

![Fig.1](https://github.com/Fantasy-AMAP/fantasy-id/blob/main/assets/method.jpg)

### Environment

```bash
git clone https://github.com/Fantasy-AMAP/fantasy-id.git
cd fantasy-id
conda create -n fantasyid python=3.10
conda activate fantasyid
pip install -r requirements.txt
```

### Download Checkpoints

The weights is available at [ModelScope](https://www.modelscope.cn/models/amap_cvlab/FantasyID/), or you can download it with the following command.

```bash
# modelscope
pip install modelscope
modelscope download --model amap_cvlab/FantasyID --local_dir Fantasy-ID
```

### Inference

```bash
python infer.py --model_path Fantasy-ID --transformer_dir Fantasy-ID --seed 42 --pcd_path ./assets/anne.ply  --img_file_path ./assets/anne.jpg --prompt "A woman in an elegant evening gown stands at a glamorous ball, her smile captivating those around her. The ballroom is grand, with crystal chandeliers casting a warm glow over the polished marble floors and intricately decorated walls. The sound of classical music fills the air, played by a live orchestra at the corner of the room. She holds a glass of champagne, her other hand lightly resting on the arm of a companion. Her movements are graceful, and she exudes confidence and sophistication, adding to the charm and elegance of the evening."
```

### Generate Point Cloud

To generate your own point cloud file of a custom image, please refer `decalib/README.md`.

## Acknowledgement

Thanks to the following open source libraries: [CogvideoX](https://github.com/THUDM/CogVideo), [DECA](https://github.com/yfeng95/DECA), [CVTHead](https://github.com/HowieMa/CVTHead), [ConsisID](https://github.com/PKU-YuanGroup/ConsisID).

## Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝.
```
@misc{zhang2025fantasyidfaceknowledgeenhanced,
      title={FantasyID: Face Knowledge Enhanced ID-Preserving Video Generation}, 
      author={Yunpeng Zhang and Qiang Wang and Fan Jiang and Yaqi Fan and Mu Xu and Yonggang Qi},
      year={2025},
      eprint={2502.13995},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2502.13995}, 
}
```
