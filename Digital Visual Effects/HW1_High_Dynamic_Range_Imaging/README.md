# VFX-HW1

## Default usage

```bash
python code/hdr.py
```

## Custom usage

```bash
# You may set argument for different experiments
# method: ['debevec', 'robertson']
# aligment: True / False
# mask_high_variance_region: True / False

python code/hdr.py \
	--method robertson \
	--alignment True \
	--mask_high_variance_region True
```

## Files structure

```
├── data
│   ├── hard ── images...
│   ├── images...
├── code
├── output # contain output images
```

## Requirements

```
matplotlib==3.2.2
numpy==1.19.5
opencv_python==4.6.0.66
```
