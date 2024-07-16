python scripts/main_tf_icon.py  --ckpt ./ckpt/v2-1_512-ema-pruned.ckpt      \
                                --root ./inputs/Real-Sketch-mask      \
                                --domain 'cross'                  \
                                --dpm_steps 20                    \
                                --dpm_order 2                   \
                                --scale 2.5                       \
                                --tau_a 0.6                       \
                                --tau_c 0.3                     \
                                --outdir ./outputs                \
                                --gpu cuda:1                    \
                                --seed 3407     \
                                --attn_mask True 
                           