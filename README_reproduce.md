# Install additional dependencies.
```
pip install fire
pip install rembg
pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
pip install invisible-watermark
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install  -U torchvision --index-url https://download.pytorch.org/whl/cu126
pip install kornia
```

You may need to modify one of the layer name in the library you installed based on the warning before this works.

```
CUDA_VISIBLE_DEVICES=0 python3 sampling.py --input_path "assets/images/street.jpg" --prompt "a vivid anime street, wind blows." --neg_prompt " " --pcd_mode "hybrid default 14 out_left_up_down" --add_index 12 --seed 1 --save_warps True --load_warps None

CUDA_VISIBLE_DEVICES=0 python3 sampling.py --input_path "assets/images/cat.jpg" --prompt "a black cat standing at the middle of a path." --neg_prompt " " --pcd_mode "rotate default 14 clockwise" --add_index 10 --seed 1 --save_warps True --load_warps None

CUDA_VISIBLE_DEVICES=0 python3 sampling.py --input_path "assets/images/waterfall.jpg" --prompt "a waterfall." --neg_prompt " " --pcd_mode "zoom 1 14 in" --add_index 10 --seed 1 --save_warps True --load_warps None

CUDA_VISIBLE_DEVICES=0 python3 sampling.py --input_path "assets/images/rocket.jpg" --prompt "a rocket rising." --neg_prompt " " --pcd_mode "tilt default 14 up" --add_index 10 --seed 1 --save_warps True --load_warps None

```

```
CUDA_VISIBLE_DEVICES=0 python3 sampling_with_depth_anything_v2.py --input_path "assets/images/street.jpg" --prompt "a vivid anime street, wind blows." --neg_prompt " " --pcd_mode "hybrid default 14 out_left_up_down" --add_index 12 --seed 1 --save_warps True --load_warps None

CUDA_VISIBLE_DEVICES=0 python3 sampling_with_depth_anything_v2.py --input_path "assets/images/cat.jpg" --prompt "a black cat standing at the middle of a path." --neg_prompt " " --pcd_mode "rotate default 14 clockwise" --add_index 10 --seed 1 --save_warps True --load_warps None

CUDA_VISIBLE_DEVICES=0 python3 sampling_with_depth_anything_v2.py --input_path "assets/images/waterfall.jpg" --prompt "a waterfall." --neg_prompt " " --pcd_mode "zoom 1 14 in" --add_index 10 --seed 1 --save_warps True --load_warps None

CUDA_VISIBLE_DEVICES=0 python3 sampling_with_depth_anything_v2.py --input_path "assets/images/rocket.jpg" --prompt "a rocket rising." --neg_prompt " " --pcd_mode "tilt default 14 up" --add_index 10 --seed 1 --save_warps True --load_warps None
```