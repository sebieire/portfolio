# Computer Vision - Feature Detection and Image Matching

A Python implementation of scale and rotation invariant feature point detection inspired by SIFT (Scale-Invariant Feature Transform), along with correlation-based image matching algorithms.

## Features

### Task 1: Feature Point Detection
- **Scale-Space Representation**: Creates 12 Gaussian smoothing kernels with increasing σ values
- **Difference of Gaussian (DoG)**: Calculates DoG images at all scales for feature detection
- **Keypoint Detection**: Identifies keypoints through thresholding and non-maxima suppression in scale-space
- **Gradient Calculation**: Computes image derivatives at all scales
- **Orientation Assignment**: Determines dominant orientation for each keypoint using gradient histograms
- **Feature Visualization**: Displays detected keypoints with scale and orientation indicators

### Task 2: Image Matching
- **Template Matching**: Implements correlation-based area matching algorithm
- **Cross-Correlation**: Calculates normalized cross-correlation between image patches
- **Match Visualization**: Identifies and displays the best matching location

## Technical Implementation

- **Multi-scale Analysis**: Uses σ = 2^(k/2) for k = 0 to 11 to create scale-space
- **Non-maxima Suppression**: 3D scale-space suppression (26 neighbors check)
- **Orientation Histogram**: 36-bin histogram for robust orientation estimation
- **Weighted Gradients**: Gaussian-weighted gradient magnitudes for orientation calculation

## Performance Highlights

- Successfully detects over 8000 keypoints in test images
- Robust scale and rotation invariant feature detection
- Accurate template matching with cross-correlation scores > 0.9
- Efficient implementation with optimized NumPy operations

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the feature detection and matching
python feature_detection.py
```

### Configuration Options

The script includes various flags to control output and execution:
- `RUN_TASKS`: Select which tasks to run ([1], [2], or [1,2])
- `OUTPUT_*_IMG`: Toggle image output display
- `OUTPUT_*_PLOT`: Toggle 3D plot visualization
- `EXECUTE_*`: Control computationally intensive operations

## Requirements

- Python 3.12+
- NumPy for numerical computations
- OpenCV for image processing operations
- Matplotlib for visualization and plotting

## Project Structure

```
Computer_Vision_Feature_Detection/
├── feature_detection.py       # Main implementation
├── sample_image_1.jpg        # Test image 1
├── sample_image_2.jpg        # Test image 2
├── result_1F_no_orientations.png  # Sample output
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## Technical Details

### Scale-Space Construction
The implementation creates a pyramid of Gaussian-smoothed images with exponentially increasing standard deviations, allowing detection of features at multiple scales.

### Keypoint Detection Algorithm
1. Compute Difference of Gaussian (DoG) images
2. Apply threshold (T=10) to identify candidate points
3. Perform non-maxima suppression in 3D scale-space
4. Extract keypoint coordinates (x, y, σ)

### Orientation Assignment
1. Calculate gradients in 7×7 neighborhood around each keypoint
2. Apply Gaussian weighting with σ = 9σ_keypoint/2
3. Build 36-bin orientation histogram
4. Select maximum bin as dominant orientation

## Author

**Sebastian Scheibe**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*This project was completed in 2020 as part of machine vision studies and has been updated for Python 3.12 compatibility.*