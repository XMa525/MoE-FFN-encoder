# MoE-FFN Encoder for Pathology Foundation Model Adaptation

This repository provides the implementation of a lightweight MoE-FFN encoder adaptation framework for pathology foundation models and downstream whole-slide image (WSI) classification.

The released pipeline focuses on target-domain adaptation. Given a target WSI dataset, users can prepare slide metadata, construct role-specific high-confidence patch candidates, build role prototypes, train a target-adapted MoE-FFN encoder, extract adapted WSI bag features, and train downstream MIL classifiers.

## Overview

Pathology foundation models provide strong generic representations, but their frozen features may still be suboptimal for specific downstream diagnostic tasks, especially under small-sample or domain-shifted settings. This project adapts selected high-level transformer FFN layers with lightweight Mixture-of-Experts (MoE) modules. The goal is to improve patch-level representation formation before WSI-level MIL aggregation.

The released code includes:

- MoE-FFN encoder adaptation modules
- Role-guided target-domain adaptation components
- Role prototype construction scripts
- Target proposal pool construction scripts
- Feature extraction scripts for WSI bags
- Downstream MIL training code

Large WSI files, extracted features, model checkpoints, and original foundation model weights are not included in this repository.

## Repository Structure

text MoE-FFN-encoder/   configs/          # Example configuration files   distillation/     # Distillation and adaptation losses   downstream/       # Downstream WSI-level feature extraction and MIL training code   models/           # Backbone wrappers, MoE-FFN modules, and MIL models   scripts/          # Role prototype construction and utility scripts   trainers/         # Target-domain encoder adaptation training code   utils/            # utility functions   requirements.txt   README.md 

## Installation

Clone this repository:

bash git clone https://github.com/XMa525/MoE-FFN-encoder.git cd MoE-FFN-encoder 

Create a Python environment and install dependencies:

bash conda create -n moe_ffn_encoder python=3.10 conda activate moe_ffn_encoder pip install -r requirements.txt 

Additional dependencies may be required for specific foundation models, such as DINOv2, OpenCLIP, UNI, Virchow/Virchow2, or other pathology foundation models. Please download the corresponding backbone weights from their official sources.

## 1. Prepare Target Dataset Metadata

The pipeline expects a slide-level CSV file describing the target WSI dataset. An example format is:

text slide_id,label,project,svs_path,h5_path,split case_001,0,BRACS,/path/to/case_001.svs,/path/to/case_001.h5,train case_002,1,BRACS,/path/to/case_002.svs,/path/to/case_002.h5,val case_003,0,BRACS,/path/to/case_003.svs,/path/to/case_003.h5,test 

The fields are:

- slide_id: unique slide identifier;
- label: WSI-level class label;
- project: dataset or project name;
- svs_path: path to the original WSI file;
- h5_path: path to the patch coordinate file;
- split: one of train, val, or test.

The h5_path file should contain patch coordinates generated during WSI preprocessing. The original WSI files and coordinate files are not provided in this repository.

## 2. Prepare High-Confidence Role Candidate CSV Files

Before target-domain adaptation, users need to prepare high-confidence patch candidates for a small set of pathology-related roles. In our experiments, we use three role types for each target task.

For example, in the BRACS task, the three roles are:

text atypical_epithelial_lesion benign_or_normal_epithelium fibrocollagenous_stroma 

Users can prepare these role candidate CSV files in different ways:

- using CONCH or other vision-language pathology models with task-specific text prompts;
- using pathologist annotations, if available;
- using high-confidence patch mining rules designed for the target dataset.

For CONCH-based labeling, users should design prompts according to the key tissue patterns in their own dataset and diagnostic task. The output should be one CSV file per role, containing high-confidence patch candidates for that role.

## 3. Build Target-Domain Role Prototypes

After preparing high-confidence candidate CSV files, build role prototypes using the provided script.

Example:

bash python scripts/build_role_prototypes_from_tcga_candidates_virchow2.py \   --role-csv atypical_epithelial_lesion=analysis_outputs/BRACS_role_candidates_3role_filtered/candidate_core_atypical_epithelial_lesion.csv \   --role-csv benign_or_normal_epithelium=analysis_outputs/BRACS_role_candidates_3role_filtered/candidate_core_benign_or_normal_epithelium.csv \   --role-csv fibrocollagenous_stroma=analysis_outputs/BRACS_role_candidates_3role_filtered/candidate_core_fibro_adipose_stroma.csv \   --output-dir analysis_outputs/BRACS_role_proto_v1 \   --batch-size 64 \   --num-workers 8 \   --target-layer 26 \   --token-pool mean \   --normalize-patch-feature \   --normalize-prototype \   --proto-agg cluster_mean \   --cluster-k 3 \   --min-cluster-size 3 \   --max-per-role 298 \   --proto-source-level patch \   --device cuda 

The output directory contains the role prototype files used in target-domain encoder adaptation.

Although the script name contains tcga_candidates, it accepts general role candidate CSV files and can be used for target-domain candidate patches.

## 4. Build Target Proposal Pool

Next, build the proposal pool used for target-domain encoder training.

Example:

bash python downstream/build_BRACS_proposal_pool.py \   --config configs/BRACS_build_fixed_can.yaml 

The proposal pool provides training candidates for target-domain MoE-FFN encoder adaptation. The exact configuration should specify the target dataset metadata, role prototype paths, candidate selection settings, and output paths.

## 5. Target-Domain MoE-FFN Encoder Adaptation

After building the role prototypes and proposal pool, run target-domain encoder adaptation.

Example:

bash python trainers/train_parotid_encoder_only.py \   --config configs/BRACS_encoder_only.yaml 

The configuration file should specify:

- target dataset split CSV;
- target proposal pool;
- role prototype path;
- backbone model type and weight path;
- MoE-FFN layer settings;
- training hyperparameters;
- output directory for the adapted encoder checkpoint.

A typical output checkpoint is:

text outputs/BRACS_encoder_only_v1/best_encoder_only_online_pool_student.pth 

## 6. Extract Adapted WSI Bag Features

After target-domain encoder adaptation, insert the trained MoE-FFN modules into the corresponding foundation model or transformer encoder and extract patch-level features for each WSI bag.

Example with UNI-MoE:

bash python downstream/extract_bag_features.py \   --slides_csv ../data/BRACS/bracs_split.csv \   --raw_dir ../data/BRACS \   --h5_dir ../data/BRACS/PATCHES/patches \   --out_dir features/BRACS/uni-moe/pt_files_transfer_v1 \   --encoder_name uni_moe \   --uni_weight ../UNI/pytorch_model.bin \   --stage2_ckpt outputs/BRACS_encoder_only_v1/best_encoder_only_online_pool_student.pth \   --target_block_1 21 \   --target_block_2 22 \   --source_stage2_layer_1 9 \   --source_stage2_layer_2 10 \   --shared_expert \   --freeze_backbone_except_moe \   --split test \   --batch_size 64 \   --max_patches 1024 

The output feature files are saved under:

text features/BRACS/uni-moe/pt_files_transfer_v1 

Each slide-level feature file contains patch-level features that can be used for downstream MIL training.

## 7. Train Downstream MIL Classifier

Finally, train a WSI-level MIL classifier using the extracted features.

Example with ABMIL:

bash python downstream/train_abmil.py \   --slides_csv ../data/BRACS/bracs_split.csv \   --feature_dir features/BRACS/uni-moe/pt_files_transfer_v1 \   --out_dir ./results/downstream/BRACS/abmil_uni_moe_transfer_v1 \   --mil_model abmil \   --att_dim 64 \   --lr 1e-4 \   --weight_decay 1e-4 \   --shuffle_instances \   --epochs 15 \   --patience 10 \   --batch_size 1 \   --num_workers 8 \   --monitor balanced_early_auc \   --early_auc_tol 0.03 \   --min_val_sens 0.30 \   --min_val_spec 0.60 \   --min_val_f1 1e-8 \   --min_select_epoch 8 \   --seed 37 

The downstream training script reports WSI-level classification metrics such as AUC, accuracy, F1-score, sensitivity, and specificity.

## Checkpoints and Pretrained Weights

This repository does not include large checkpoints, extracted WSI features, or original foundation model weights.

- Original foundation model weights should be downloaded from their official sources.
- Lightweight MoE-FFN adapted encoder checkpoints may be released separately.
- Extracted features for public datasets may be released separately when allowed by the dataset license.

## Notes on Private Data

Some experiments may involve private pathology datasets. These datasets cannot be redistributed. For private datasets, this repository provides only the expected input format and running pipeline.

## Citation



## License

This repository is released for research use. Please also follow the licenses of all external foundation models and datasets used in your experiments.