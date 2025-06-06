# VFX-HW2

## Our Approach

1. Wrapping: Wrap images into cylindrical coordinate.
2. Feature detection + description: We implement SIFT feature detection!
3. Feature matching: We use ANN to speed up matching!
4. Alignment: Pairwise alignment + End-to-End alignment.
5. Image blending

## Default usage

```bash
python code/main.py
```

## Files structure

```
├── data
│   ├── images...
├── code
```

## Requirements dependency

```
annoy==1.17.1
match==0.3.2
numpy==1.19.5
opencv_python==4.6.0.66
scikit_image==0.15.0
scipy==1.3.1
skimage==0.0
```
