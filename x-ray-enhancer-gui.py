import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from skimage.restoration import denoise_tv_chambolle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QSlider, QComboBox, QPushButton, QFileDialog, QGroupBox,
                            QGridLayout, QFrame, QListWidget, QAbstractItemView, QDoubleSpinBox,
                            QSpinBox, QCheckBox, QSplitter, QMessageBox, QListWidgetItem, QScrollArea,QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QDrag, QIcon


class DentalXRayProcessor:
    """
    A modular pipeline-based processor for dental X-ray images.
    Each processing step is a separate operation that can be added to the pipeline.
    """
    
    def __init__(self):
        self.pipeline = []
        self.operations = {
            "nlm_denoise": self.nlm_denoise,
            "bilateral_filter": self.bilateral_filter,
            "adjust_contrast_brightness": self.adjust_contrast_brightness,
            "gamma_correction": self.gamma_correction,
            "clahe_enhancement": self.clahe_enhancement,
            "edge_enhancement": self.edge_enhancement,
            "median_blur": self.median_blur,
            "unsharp_mask": self.unsharp_mask,
            "edge_aware_smoothing": self.edge_aware_smoothing,
            "tv_denoise": self.tv_denoise,
            "advanced_edge_enhancement": self.advanced_edge_enhancement,
            "overlay_original": self.overlay_original,
            "invert": self.invert,
        }
        
        # parameter specs for each operation
        self.operation_params = {
            "nlm_denoise": {
                "h": {"type": "double", "min": 1, "max": 30, "default": 10, "decimals": 3, 
                      "description": "Filter strength (higher = more smoothing)"},
                "template_window": {"type": "int", "min": 3, "max": 15, "default": 7, "step": 2,
                                  "description": "Template window size (must be odd)"},
                "search_window": {"type": "int", "min": 5, "max": 35, "default": 21, "step": 2,
                                "description": "Search window size (must be odd)"}
            },
            "bilateral_filter": {
                "d": {"type": "int", "min": 1, "max": 20, "default": 9, 
                     "description": "Diameter of pixel neighborhood"},
                "sigma_color": {"type": "double", "min": 1, "max": 150, "default": 75, "decimals": 3,
                              "description": "Filter sigma in color space"},
                "sigma_space": {"type": "double", "min": 1, "max": 150, "default": 75, "decimals": 3,
                              "description": "Filter sigma in coordinate space"}
            },
            "adjust_contrast_brightness": {
                "contrast": {"type": "double", "min": 0.1, "max": 3.0, "default": 1.0, "decimals": 3,
                           "description": "Contrast factor (1.0 is neutral)"},
                "brightness": {"type": "int", "min": -50, "max": 100, "default": 0,
                             "description": "Brightness adjustment"}
            },
            "gamma_correction": {
                "gamma": {"type": "double", "min": 0.1, "max": 3.0, "default": 1.0, "decimals": 3,
                        "description": "Gamma value (1.0 is neutral)"}
            },
            "clahe_enhancement": {
                "clip_limit": {"type": "double", "min": 0.5, "max": 10.0, "default": 2.0, "decimals": 3,
                             "description": "Threshold for contrast limiting"},
                "tile_grid_size": {"type": "int", "min": 2, "max": 16, "default": 8,
                                 "description": "Size of grid for histogram equalization"}
            },
            "edge_enhancement": {
                "strength": {"type": "double", "min": 0.1, "max": 2.0, "default": 0.5, "decimals": 3,
                           "description": "Edge enhancement strength"}
            },
            "median_blur": {
                "ksize": {"type": "int", "min": 3, "max": 15, "default": 3, "step": 2,
                        "description": "Kernel size (must be odd)"}
            },
            "unsharp_mask": {
                "amount": {"type": "double", "min": 0.1, "max": 3.0, "default": 1.0, "decimals": 3,
                         "description": "Sharpening strength"},
                "radius": {"type": "int", "min": 1, "max": 21, "default": 5, "step": 2,
                         "description": "Blur radius for the mask (must be odd)"}
            },
            "edge_aware_smoothing": {
                "edge_threshold": {"type": "int", "min": 10, "max": 100, "default": 30,
                                 "description": "Threshold for edge detection"},
                "edge_dilate": {"type": "int", "min": 1, "max": 5, "default": 1,
                              "description": "Dilation iterations for edge mask"}
            },
            "tv_denoise": {
                "weight": {"type": "double", "min": 0.01, "max": 1.0, "default": 0.1, "decimals": 3,
                         "description": "Denoising weight (higher = more smoothing)"},
                "eps": {"type": "double", "min": 1e-5, "max": 1e-2, "default": 2e-4, "decimals": 5,
                       "description": "Relative difference threshold for stopping"},
                "max_iter": {"type": "int", "min": 50, "max": 500, "default": 200,
                           "description": "Maximum number of iterations"}
            },
            "advanced_edge_enhancement": {
                "edge_threshold": {"type": "int", "min": 10, "max": 100, "default": 30,
                                "description": "Threshold for edge detection (lower = more sensitive)"},
                "detail_preservation": {"type": "double", "min": 0.1, "max": 1.0, "default": 0.7, "decimals": 3,
                                    "description": "How much detail to preserve in edges (0-1)"},
                "enhancement_strength": {"type": "double", "min": 0.5, "max": 100.0, "default": 1.2, "decimals": 3,
                                    "description": "Strength of edge enhancement"},
                "smoothing_factor": {"type": "double", "min": 0.1, "max": 1.0, "default": 0.3, "decimals": 3,
                                    "description": "Amount of smoothing to apply to non-edge areas"}
            },
            "overlay_original": {
                "opacity": {"type": "double", "min": 0.0, "max": 1.0, "default": 0.5, "decimals": 2,
                        "description": "Opacity of the original image overlay (0-1)"},
                "blend_mode": {"type": "combo", "options": ["normal", "multiply", "add", "subtract", 
                                                        "overlay", "soft_light", "hard_light", "difference"],
                            "default": "normal", 
                            "description": "Method used to blend the original image"}
            },
            "invert": {
                "strength": {"type": "double", "min": 0.0, "max": 1.0, "default": 1.0, "decimals": 2,
                            "description": "Strength of the inversion effect (0-1)"}
            },
        }
        
        # names for operations
        self.operation_names = {
            "nlm_denoise": "Non-Local Means Denoising",
            "bilateral_filter": "Bilateral Filter",
            "adjust_contrast_brightness": "Adjust Contrast & Brightness",
            "gamma_correction": "Gamma Correction",
            "clahe_enhancement": "CLAHE Enhancement",
            "edge_enhancement": "Edge Enhancement",
            "median_blur": "Median Blur",
            "unsharp_mask": "Unsharp Mask",
            "edge_aware_smoothing": "Edge-Aware Smoothing",
            "tv_denoise": "Total Variation Denoising",
            "advanced_edge_enhancement": "Advanced Edge Enhancement",
            "overlay_original": "Overlay Original Image",
            "invert": "Invert Image",
        }
    
    def add_operation(self, operation_name, params=None):
        """Add an operation to the processing pipeline."""
        if operation_name not in self.operations:
            raise ValueError(f"Unknown operation: {operation_name}")
        
        # default parameters if none are provided
        if params is None:
            params = {}
            for param_name, param_spec in self.operation_params[operation_name].items():
                params[param_name] = param_spec["default"]
        
        self.pipeline.append({
            "operation": operation_name,
            "params": params
        })
        
        return self  # Enable method chaining
    
    def set_original_image(self, image):
        """Set the original image to use for overlay operations."""
        if image is not None:
            self.original_image = image.copy()
            
    def process(self, image):
        """Process an image through the entire pipeline."""
        # Store for overlay operations
        self.set_original_image(image)
        
        result = image.copy()
        
        for step in self.pipeline:
            operation_name = step["operation"]
            params = step["params"]
            
            # call operation function with the current image and parameters
            result = self.operations[operation_name](result, **params)
            
        return result
    
    def clear_pipeline(self):
        """Clear the processing pipeline."""
        self.pipeline = []
        return self
    
    def set_pipeline_from_json(self, json_pipeline):
        """Set the pipeline from a JSON string or dictionary."""
        if isinstance(json_pipeline, str):
            pipeline_data = json.loads(json_pipeline)
        else:
            pipeline_data = json_pipeline
            
        self.clear_pipeline()
        for step in pipeline_data:
            self.add_operation(step["operation"], step.get("params", {}))
            
        return self
    
    def get_pipeline_as_json(self):
        """Get the current pipeline as a JSON string."""
        return json.dumps(self.pipeline, indent=2)
    
    # ===== Operation Methods =====
    
    def nlm_denoise(self, image, h=10, template_window=7, search_window=21):
        """
        Apply Non-Local Means denoising to reduce grain.
        
        Args:
            image: Input image
            h: Filter strength parameter
            template_window: Size of template window
            search_window: Size of search window
        """
        # Ensure parameters are valid
        template_window = int(template_window)
        if template_window % 2 == 0:
            template_window += 1
            
        search_window = int(search_window)
        if search_window % 2 == 0:
            search_window += 1
            
        return cv2.fastNlMeansDenoising(
            image, None, h=h,
            templateWindowSize=template_window,
            searchWindowSize=search_window
        )
    
    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filtering to reduce noise while preserving edges.
        
        Args:
            image: Input image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
        """
        d = int(d)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def adjust_contrast_brightness(self, image, contrast=1.0, brightness=0):
        """
        Adjust contrast and brightness.
        
        Args:
            image: Input image
            contrast: Contrast factor (1.0 is neutral)
            brightness: Brightness adjustment
        """
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    def gamma_correction(self, image, gamma=1.0):
        """
        Apply gamma correction.
        
        Args:
            image: Input image
            gamma: Gamma value (1.0 is neutral)
        """
        lut = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, lut)
    
    def clahe_enhancement(self, image, clip_limit=2.0, tile_grid_size=8):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization (int or [int, int])
        """
        # Handle tile_grid_size as either int or list
        if isinstance(tile_grid_size, list):
            # Use the first element if it's a list
            grid_size = int(tile_grid_size[0])
        else:
            grid_size = int(tile_grid_size)
            
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        return clahe.apply(image)
    
    def edge_enhancement(self, image, strength=0.5):
        """
        Enhance edges using Laplacian.
        
        Args:
            image: Input image
            strength: Edge enhancement strength
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
        laplacian_abs = np.absolute(laplacian)
        laplacian_max = np.max(laplacian_abs)
        
        if laplacian_max > 0:  # division by zero
            laplacian_norm = np.uint8(255 * laplacian_abs / laplacian_max)
            return cv2.addWeighted(image, 1.0, laplacian_norm, strength, 0)
        
        return image
    
    def median_blur(self, image, ksize=3):
        """
        Apply median blur for noise reduction.
        
        Args:
            image: Input image
            ksize: Kernel size (must be odd)
        """
        #  kernel size should be is odd
        ksize = int(ksize)
        if ksize % 2 == 0:
            ksize += 1
            
        return cv2.medianBlur(image, ksize)
    
    def unsharp_mask(self, image, amount=1.0, radius=5):
        """
        Apply unsharp mask for sharpening.
        
        Args:
            image: Input image
            amount: Sharpening strength
            radius: Blur radius for the mask
        """
        # Ensure radius is odd
        radius = int(radius)
        if radius % 2 == 0:
            radius += 1
            
        gaussian = cv2.GaussianBlur(image, (radius, radius), 0)
        return cv2.addWeighted(image, 1.0 + amount, gaussian, -amount, 0)
    
    def invert(self, image, strength=1.0):
        """
        Invert the image (negative effect).
        
        Args:
            image: Input image
            strength: Strength of the inversion effect (0-1)
        """
        # Full inversion
        inverted = 255 - image
        
        # Blend with original based on strength
        if strength < 1.0:
            return cv2.addWeighted(image, 1.0 - strength, inverted, strength, 0)
        else:
            return inverted

    def edge_aware_smoothing(self, image, edge_threshold=30, edge_dilate=1):
        """
        Apply edge-aware smoothing (preserves edges, smooths non-edge areas).
        
        Args:
            image: Input image
            edge_threshold: Threshold for edge detection
            edge_dilate: Dilation iterations for edge mask
        """
        # Detect edges
        edge_threshold = int(edge_threshold)
        edge_dilate = int(edge_dilate)
        
        edges = cv2.Canny(image, edge_threshold, edge_threshold * 3)
        
        # Dilate edges to protect areas near edges
        dilated_edges = cv2.dilate(
            edges, 
            np.ones((3,3), np.uint8), 
            iterations=edge_dilate
        )
        
        # Create inverse mask (1 for non-edge areas, 0 for edges)
        smooth_mask = 1 - (dilated_edges / 255.0)
        
        # Apply median blur to the whole image
        median_filtered = cv2.medianBlur(image, 3)
        
        # Combine original edges with median-filtered non-edge areas
        return np.uint8(image * (1 - smooth_mask) + median_filtered * smooth_mask)
        
    def advanced_edge_enhancement(self, image, edge_threshold=30, detail_preservation=0.7, enhancement_strength=1.2, smoothing_factor=0.3):
        """
        Advanced edge enhancement that targets dental structures while avoiding noise.
        
        Args:
            image: Input image
            edge_threshold: Threshold for edge detection (lower = more sensitive)
            detail_preservation: How much detail to preserve (0-1)
            enhancement_strength: Strength of edge enhancement
            smoothing_factor: Amount of smoothing to apply to non-edge areas
        """
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Step 1: Initial edge detection using Canny
        edges_canny = cv2.Canny(image, edge_threshold, edge_threshold * 2)
        
        # Step 2: Get gradient magnitude for edge strength map
        grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = cv2.magnitude(grad_x, grad_y)
        
        # Normalize gradient magnitude to 0-1
        if gradient_mag.max() > 0:
            gradient_mag = gradient_mag / gradient_mag.max()
        
        # Step 3: Create a mask of significant edges (combining Canny with gradient magnitude)
        _, strong_edges = cv2.threshold(
            (gradient_mag * 255).astype(np.uint8), 
            edge_threshold, 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Combine Canny edges with gradient-based edges
        combined_edges = cv2.bitwise_or(edges_canny, strong_edges)
        
        # Step 4: Dilate edges slightly to include nearby areas
        edge_kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(combined_edges, edge_kernel, iterations=1)
        
        # Create edge mask (1 for edges, 0 for non-edges)
        edge_mask = dilated_edges.astype(np.float32) / 255.0
        
        # Step 5: Apply bilateral filter to create a detail-preserved but smoothed base image
        detail_preserved = cv2.bilateralFilter(
            image, 
            d=5, 
            sigmaColor=30 * (1 - detail_preservation + 0.1),  # Lower value = more details
            sigmaSpace=30 * (1 - detail_preservation + 0.1)
        ).astype(np.float32)
        
        # Step 6: Apply a stronger smoothing to non-edge areas
        smoothed = cv2.GaussianBlur(
            image, 
            (0, 0), 
            sigmaX=smoothing_factor * 10,
            sigmaY=smoothing_factor * 10
        ).astype(np.float32)
        
        # Step 7: Create a base image that preserves details in edge areas and smooths non-edge areas
        base = detail_preserved * edge_mask + smoothed * (1 - edge_mask)
        
        # Step 8: Extract and enhance edges
        high_pass = image.astype(np.float32) - base
        enhanced_edges = high_pass * enhancement_strength
        
        # Step 9: Recombine the enhanced edges with the base
        result = base + enhanced_edges
        
        # Ensure the result is within valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

    def tv_denoise(self, image, weight=0.1, eps=2e-4, max_iter=200):
        """
        Apply Total Variation denoising using Chambolle's algorithm.
        
        Args:
            image: Input image
            weight: Denoising weight (higher = more smoothing)
            eps: Relative difference threshold for stopping
            max_iter: Maximum number of iterations
        """
        # Convert uint8 image to float [0, 1]
        img_float = image.astype(np.float32) / 255.0
        
        # Apply TV denoising
        denoised = denoise_tv_chambolle(
            img_float,
            weight=weight,
            eps=eps,
            max_num_iter=int(max_iter),
            channel_axis=None  # For grayscale
        )
            
        # Convert back to uint8
        return np.uint8(denoised * 255)

    def overlay_original(self, image, opacity=0.5, blend_mode="normal"):
        """
        Overlay the original image onto the current image with various blend modes.
        
        Args:
            image: Current processed image
            opacity: Opacity of the original image (0-1)
            blend_mode: Blending mode ('normal', 'multiply', 'add', 'subtract', 'soft_light', 'hard_light', etc.)
        """
        # Store the original image as a class attribute if it doesn't exist yet
        if not hasattr(self, 'original_image') or self.original_image is None:
            self.original_image = image.copy()
            return image
        
        # Ensure images are the same size
        if self.original_image.shape != image.shape:
            # Resize original to match current if needed
            self.original_image = cv2.resize(self.original_image, (image.shape[1], image.shape[0]))
        
        # Convert to float for blending
        img_float = image.astype(np.float32)
        orig_float = self.original_image.astype(np.float32)
        
        # Apply the selected blend mode
        if blend_mode == "normal":
            # Simple alpha blending
            blended = img_float * (1 - opacity) + orig_float * opacity
        
        elif blend_mode == "multiply":
            # Multiply blend mode (darkens image)
            blended = (img_float * orig_float) / 255.0
            blended = img_float * (1 - opacity) + blended * opacity
        
        elif blend_mode == "add":
            # Add blend mode (lightens image)
            blended = img_float + (orig_float * opacity)
            
        elif blend_mode == "subtract":
            # Subtract blend mode (creates negative effect)
            blended = img_float - (orig_float * opacity)
        
        elif blend_mode == "soft_light":
            # Soft light (similar to overlay, but softer)
            def soft_light(a, b):
                return ((1 - 2*b/255.0) * (a**2)/255.0 + 2*b*a/255.0)
            
            blended = soft_light(img_float, orig_float)
            blended = img_float * (1 - opacity) + blended * opacity
        
        elif blend_mode == "hard_light":
            # Hard light (similar to overlay, but harder)
            def hard_light(a, b):
                mask = b > 127.5
                result = np.zeros_like(a, dtype=np.float32)
                result[mask] = 2 * a[mask] * b[mask] / 255.0
                result[~mask] = 255.0 - 2 * (255.0 - a[~mask]) * (255.0 - b[~mask]) / 255.0
                return result
                
            blended = hard_light(img_float, orig_float)
            blended = img_float * (1 - opacity) + blended * opacity
        
        elif blend_mode == "overlay":
            # Overlay (combination of multiply and screen)
            def overlay(a, b):
                mask = a > 127.5
                result = np.zeros_like(a, dtype=np.float32)
                result[mask] = 255.0 - 2 * (255.0 - a[mask]) * (255.0 - b[mask]) / 255.0
                result[~mask] = 2 * a[~mask] * b[~mask] / 255.0
                return result
                
            blended = overlay(img_float, orig_float)
            blended = img_float * (1 - opacity) + blended * opacity
        
        elif blend_mode == "difference":
            # Difference (subtracts darker from lighter)
            blended = np.abs(img_float - orig_float)
            blended = img_float * (1 - opacity) + blended * opacity
        
        else:
            # Default to normal if blend mode not recognized
            blended = img_float * (1 - opacity) + orig_float * opacity
        
        # Ensure the result is within valid range
        return np.clip(blended, 0, 255).astype(np.uint8)

class ProcessingThread(QThread):
    """Thread for processing the image in the background."""
    processingFinished = pyqtSignal(np.ndarray)
    
    def __init__(self, processor, image):
        super().__init__()
        self.processor = processor
        self.image = image
        
    def run(self):
        """Process the image and emit the result."""
        result = self.processor.process(self.image)
        self.processingFinished.emit(result)


class ReorderableListWidget(QListWidget):
    """Custom list widget that supports drag-and-drop reordering."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        
    def dropEvent(self, event):
        """Handle drop events for reordering."""
        super().dropEvent(event)
        # Notify parent that items have been reordered
        self.parent().on_pipeline_reordered()

class FullScreenImageDialog(QDialog):
    """Dialog to display an image in full screen."""
    
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Full Screen Image")
        self.setWindowState(Qt.WindowFullScreen)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Display the image
        self.display_image()
        
        # Instructions label
        instructions = QLabel("Press ESC or click anywhere to exit")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 128); padding: 5px;")
        layout.addWidget(instructions)
        
    def display_image(self):
        if self.image is not None:
            # Convert to QImage
            height, width = self.image.shape
            bytes_per_line = width
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # Convert to QPixmap and scale to fit screen
            pixmap = QPixmap.fromImage(q_img)
            screen_size = QApplication.desktop().screenGeometry().size()
            scaled_pixmap = pixmap.scaled(screen_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.image_label.setPixmap(scaled_pixmap)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
    
    def mousePressEvent(self, event):
        self.close()


class DentalXRayEnhancerGUI(QMainWindow):
    """
    GUI for dental X-ray enhancement with configurable pipeline.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize the processor
        self.processor = DentalXRayProcessor()
        
        # Store the original image and processed image
        self.original_image = None
        self.processed_image = None
        
        # Initialize the UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Dental X-Ray Enhancement Pipeline")
        self.setMinimumSize(1200, 800)
        
        # central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # main splitter to divide control panel and image preview
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(400)
        left_panel.setMaximumWidth(500)
        
        # right panel for image preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # panels to the splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 800])
        
        # file controls group
        file_group = QGroupBox("File Operations")
        file_layout = QGridLayout(file_group)
        
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_image_btn, 0, 0)
        
        self.save_image_btn = QPushButton("Save Processed Image")
        self.save_image_btn.clicked.connect(self.save_processed_image)
        self.save_image_btn.setEnabled(False)
        file_layout.addWidget(self.save_image_btn, 0, 1)
        
        self.load_pipeline_btn = QPushButton("Load Pipeline")
        self.load_pipeline_btn.clicked.connect(self.load_pipeline)
        file_layout.addWidget(self.load_pipeline_btn, 1, 0)
        
        self.save_pipeline_btn = QPushButton("Save Pipeline")
        self.save_pipeline_btn.clicked.connect(self.save_pipeline)
        file_layout.addWidget(self.save_pipeline_btn, 1, 1)
        
        left_layout.addWidget(file_group)
        
        # pipeline controls group
        pipeline_group = QGroupBox("Pipeline Operations")
        pipeline_layout = QVBoxLayout(pipeline_group)
        
        # operation selection
        operation_layout = QHBoxLayout()
        operation_layout.addWidget(QLabel("Add Operation:"))
        
        self.operation_combo = QComboBox()
        # Populate with available operations
        for op_name, op_display_name in self.processor.operation_names.items():
            self.operation_combo.addItem(op_display_name, op_name)
        operation_layout.addWidget(self.operation_combo, 1)
        
        self.add_operation_btn = QPushButton("Add")
        self.add_operation_btn.clicked.connect(self.add_operation)
        operation_layout.addWidget(self.add_operation_btn)
        
        pipeline_layout.addLayout(operation_layout)
        
        # Pipeline list
        pipeline_layout.addWidget(QLabel("Current Pipeline:"))
        
        self.pipeline_list = ReorderableListWidget(self)
        self.pipeline_list.currentRowChanged.connect(self.show_operation_parameters)
        self.pipeline_list.setMinimumHeight(150)
        
        pipeline_list_layout = QHBoxLayout()
        pipeline_list_layout.addWidget(self.pipeline_list)
        
        # Buttons for pipeline manipulation
        pipeline_buttons_layout = QVBoxLayout()
        
        self.remove_operation_btn = QPushButton("Remove")
        self.remove_operation_btn.clicked.connect(self.remove_operation)
        self.remove_operation_btn.setEnabled(False)
        pipeline_buttons_layout.addWidget(self.remove_operation_btn)
        
        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_operation_up)
        self.move_up_btn.setEnabled(False)
        pipeline_buttons_layout.addWidget(self.move_up_btn)
        
        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_operation_down)
        self.move_down_btn.setEnabled(False)
        pipeline_buttons_layout.addWidget(self.move_down_btn)
        
        self.clear_pipeline_btn = QPushButton("Clear All")
        self.clear_pipeline_btn.clicked.connect(self.clear_pipeline)
        pipeline_buttons_layout.addWidget(self.clear_pipeline_btn)
        
        pipeline_buttons_layout.addStretch()
        
        pipeline_list_layout.addLayout(pipeline_buttons_layout)
        pipeline_layout.addLayout(pipeline_list_layout)
        
        left_layout.addWidget(pipeline_group)
        
        # Parameter editing group
        self.param_group = QGroupBox("Operation Parameters")
        self.param_layout = QVBoxLayout(self.param_group)
        
        # a scroll area for parameters
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setFrameShape(QFrame.NoFrame)
        
        self.param_widget = QWidget()
        self.param_widget.setLayout(QVBoxLayout())
        param_scroll.setWidget(self.param_widget)
        
        self.param_layout.addWidget(param_scroll)
        left_layout.addWidget(self.param_group)
        
        # Process button
        self.process_btn = QPushButton("Process Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        left_layout.addWidget(self.process_btn)
        
        # Status label
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)
        
        # Image preview
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(QLabel("Image Preview:"))
        
        image_preview_layout = QHBoxLayout()
        
        # Original image
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(300, 300)
        self.original_label.setScaledContents(False)
        self.original_label.setFrameShape(QFrame.Box)
        original_layout.addWidget(self.original_label)
        image_preview_layout.addWidget(original_group)
        
        # Processed image
        processed_group = QGroupBox("Processed Image")
        processed_layout = QVBoxLayout(processed_group)
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(300, 300)
        self.processed_label.setScaledContents(False)
        self.processed_label.setFrameShape(QFrame.Box)
        processed_layout.addWidget(self.processed_label)
        
        # Add button controls for processed image
        processed_buttons_layout = QHBoxLayout()
        
        self.full_screen_btn = QPushButton("Full Screen")
        self.full_screen_btn.setIcon(QIcon.fromTheme("view-fullscreen"))
        self.full_screen_btn.clicked.connect(self.show_full_screen)
        self.full_screen_btn.setEnabled(False)
        processed_buttons_layout.addWidget(self.full_screen_btn)
        
        self.save_now_btn = QPushButton("Save Now")
        self.save_now_btn.setIcon(QIcon.fromTheme("document-save"))
        self.save_now_btn.clicked.connect(self.save_processed_image)
        self.save_now_btn.setEnabled(False)
        processed_buttons_layout.addWidget(self.save_now_btn)
        
        processed_layout.addLayout(processed_buttons_layout)
        image_preview_layout.addWidget(processed_group)
        
        preview_layout.addLayout(image_preview_layout)
        right_layout.addLayout(preview_layout)
        
        # Add default pipeline with edge-focused parameters
        self.add_default_pipeline()
        
        # Show the window
        self.show()
    
    def show_full_screen(self):
        """Show the processed image in full screen."""
        if self.processed_image is not None:
            full_screen_dialog = FullScreenImageDialog(self.processed_image, self)
            full_screen_dialog.exec_()  # This will work now with QDialog


    def add_default_pipeline(self):
        """Add a default edge-focused pipeline with TV denoising."""
        edge_pipeline = [
            {
                "operation": "tv_denoise",
                "params": {"weight": 0.15, "eps": 2e-4, "max_iter": 200}
            },
            {
                "operation": "nlm_denoise",
                "params": {"h": 8, "template_window": 5, "search_window": 15}
            },
            {
                "operation": "adjust_contrast_brightness",
                "params": {"contrast": 1.1, "brightness": 15}
            },
            {
                "operation": "gamma_correction",
                "params": {"gamma": 1.1}
            },
            {
                "operation": "clahe_enhancement",
                "params": {"clip_limit": 4.0, "tile_grid_size": 8}
            },
            {
                "operation": "advanced_edge_enhancement",
                "params": {"edge_threshold": 25, "detail_preservation": 0.8, "enhancement_strength": 1.5, "smoothing_factor": 0.3}
            },
            {
                "operation": "bilateral_filter",
                "params": {"d": 5, "sigma_color": 25, "sigma_space": 25}
            },
            {
                "operation": "unsharp_mask",
                "params": {"amount": 1.5, "radius": 3}
            }
        ]
        
        self.processor.set_pipeline_from_json(edge_pipeline)
        self.update_pipeline_list()
    
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.bmp *.tif);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load the image
                self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if self.original_image is None:
                    QMessageBox.critical(self, "Error", "Failed to load image.")
                    return
                
                # Display the original image
                self.display_image(self.original_image, self.original_label)
                
                # Enable processing
                self.process_btn.setEnabled(True)
                
                # Set status
                self.status_label.setText(f"Loaded image: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
    
    def save_processed_image(self):
        """save the processed image with lossless or highest quality settings."""
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to save.")
            return
                
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Image", "", 
            "TIFF Files (*.tif);;PNG Files (*.png);;BMP Files (*.bmp);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    cv2.imwrite(file_path, self.processed_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    self.status_label.setText(f"Saved processed image as JPEG (highest quality)")
                    
                elif file_path.lower().endswith('.png'):
                    cv2.imwrite(file_path, self.processed_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    self.status_label.setText(f"Saved processed image as lossless PNG")
                    
                elif file_path.lower().endswith(('.tif', '.tiff')):
                    cv2.imwrite(file_path, self.processed_image)
                    self.status_label.setText(f"Saved processed image as lossless TIFF")
                    
                elif file_path.lower().endswith('.bmp'):
                    cv2.imwrite(file_path, self.processed_image)
                    self.status_label.setText(f"Saved processed image as uncompressed BMP")
                    
                else:
                    cv2.imwrite(file_path, self.processed_image)
                    self.status_label.setText(f"Saved processed image")
                    
                self.status_label.setText(f"{self.status_label.text()} to: {os.path.basename(file_path)}")
                
                # file size information
                file_size_bytes = os.path.getsize(file_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
                self.status_label.setText(f"{self.status_label.text()} (Size: {file_size_mb:.2f} MB)")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving image: {str(e)}")

    
    def load_pipeline(self):
        """Load a pipeline from a JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    pipeline_json = f.read()
                
                self.processor.set_pipeline_from_json(pipeline_json)
                self.update_pipeline_list()
                
                self.status_label.setText(f"Loaded pipeline from: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading pipeline: {str(e)}")
    
    def save_pipeline(self):
        """Save the current pipeline to a JSON file."""
        if not self.processor.pipeline:
            QMessageBox.warning(self, "Warning", "Pipeline is empty. Nothing to save.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.processor.get_pipeline_as_json())
                
                self.status_label.setText(f"Saved pipeline to: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving pipeline: {str(e)}")
    
    def add_operation(self):
        """Add the selected operation to the pipeline."""
        operation_index = self.operation_combo.currentIndex()
        if operation_index < 0:
            return
            
        operation_id = self.operation_combo.itemData(operation_index)
        
        # Create with default parameters
        self.processor.add_operation(operation_id)
        
        # Update the list
        self.update_pipeline_list()
        
        # Select the new operation
        self.pipeline_list.setCurrentRow(self.pipeline_list.count() - 1)
    
    def remove_operation(self):
        """Remove the selected operation from the pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row < 0:
            return
            
        # Remove from processor pipeline
        del self.processor.pipeline[current_row]
        
        # Update the list
        self.update_pipeline_list()
        
        # Update button states
        self.update_button_states()
    
    def move_operation_up(self):
        """Move the selected operation up in the pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row <= 0:
            return
            
        # Swap in processor pipeline
        self.processor.pipeline[current_row], self.processor.pipeline[current_row-1] = \
            self.processor.pipeline[current_row-1], self.processor.pipeline[current_row]
        
        # Update the List
        self.update_pipeline_list()
        
        # keep selection
        self.pipeline_list.setCurrentRow(current_row - 1)
    
    def move_operation_down(self):
        """Move the selected operation down in the pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row < 0 or current_row >= len(self.processor.pipeline) - 1:
            return
            
        # Swap in processor pipeline
        self.processor.pipeline[current_row], self.processor.pipeline[current_row+1] = \
            self.processor.pipeline[current_row+1], self.processor.pipeline[current_row]
        
        # Update the list
        self.update_pipeline_list()
        
        # Keep selection
        self.pipeline_list.setCurrentRow(current_row + 1)
    
    def clear_pipeline(self):
        """Clear the entire pipeline."""
        reply = QMessageBox.question(
            self, "Confirm Clear", 
            "Are you sure you want to clear the entire pipeline?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.processor.clear_pipeline()
            self.update_pipeline_list()
            self.update_button_states()
    
    def on_pipeline_reordered(self):
        """Handle pipeline reordering from drag and drop."""
        # Rebuild the processor pipeline based on the new order
        new_pipeline = []
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            op_index = item.data(Qt.UserRole)
            new_pipeline.append(self.processor.pipeline[op_index])
        
        self.processor.pipeline = new_pipeline
        self.update_pipeline_list()
    
    def update_pipeline_list(self):
        """Update the pipeline list widget to reflect the processor pipeline."""
        self.pipeline_list.clear()
        
        for i, step in enumerate(self.processor.pipeline):
            op_name = step["operation"]
            display_name = self.processor.operation_names[op_name]
            
            # Create list item with operation name
            item = QListWidgetItem(f"{i+1}. {display_name}")
            item.setData(Qt.UserRole, i)  # Store index for reordering
            
            self.pipeline_list.addItem(item)
        
        # uupdate button states
        self.update_button_states()
    
    def update_button_states(self):
        """Update the enabled state of buttons based on selection and pipeline state."""
        has_selection = self.pipeline_list.currentRow() >= 0
        has_pipeline = len(self.processor.pipeline) > 0
        not_first = self.pipeline_list.currentRow() > 0
        not_last = self.pipeline_list.currentRow() < len(self.processor.pipeline) - 1
        
        self.remove_operation_btn.setEnabled(has_selection)
        self.move_up_btn.setEnabled(has_selection and not_first)
        self.move_down_btn.setEnabled(has_selection and not_last)
        self.clear_pipeline_btn.setEnabled(has_pipeline)
        self.save_pipeline_btn.setEnabled(has_pipeline)
    
    def show_operation_parameters(self, row):
        """Show parameters for the selected operation."""
        # clear previous parameters
        self.clear_parameter_widgets()
        
        if row < 0 or row >= len(self.processor.pipeline):
            return
            
        # Get the Operation
        operation = self.processor.pipeline[row]
        op_name = operation["operation"]
        params = operation["params"]
        
        #  parameter widgets
        param_specs = self.processor.operation_params[op_name]
        
        #  new parameter widgets
        for param_name, param_spec in param_specs.items():
            param_layout = QHBoxLayout()
            
            # Parameter label with description
            label = QLabel(f"{param_name}:")
            label.setToolTip(param_spec["description"])
            param_layout.addWidget(label)
            
            # Value widget depends on parameter type
            if param_spec["type"] == "double":
                spinner = QDoubleSpinBox()
                spinner.setMinimum(param_spec["min"])
                spinner.setMaximum(param_spec["max"])
                if "decimals" in param_spec:
                    spinner.setDecimals(param_spec["decimals"])
                    
                # Handle list values for double parameters
                param_value = params.get(param_name, param_spec["default"])
                if isinstance(param_value, list) and len(param_value) > 0:
                    param_value = float(param_value[0])
                    
                spinner.setValue(param_value)
                spinner.valueChanged.connect(
                    lambda value, param_name=param_name, row=row: 
                    self.update_parameter(row, op_name, param_name, value)
                )
                param_layout.addWidget(spinner)
            
            elif param_spec["type"] == "combo":
                combo = QComboBox()
                for option in param_spec["options"]:
                    combo.addItem(option)
                
                # Set current value
                param_value = params.get(param_name, param_spec["default"])
                index = combo.findText(param_value)
                if index >= 0:
                    combo.setCurrentIndex(index)
                
                combo.currentTextChanged.connect(
                    lambda value, op=op_name, param=param_name, row=row: 
                    self.update_parameter(row, op, param, value)
                )
                param_layout.addWidget(combo)
                
            elif param_spec["type"] == "int":
                spinner = QSpinBox()
                spinner.setMinimum(param_spec["min"])
                spinner.setMaximum(param_spec["max"])
                if "step" in param_spec:
                    spinner.setSingleStep(param_spec["step"])
                    
                # Handle list values for int parameters
                param_value = params.get(param_name, param_spec["default"])
                if isinstance(param_value, list) and len(param_value) > 0:
                    param_value = int(param_value[0])
                    
                spinner.setValue(param_value)
                spinner.valueChanged.connect(
                    lambda value, param_name=param_name, row=row: 
                    self.update_parameter(row, operation, param_name, value)
                )
                param_layout.addWidget(spinner)
            
            self.param_widget.layout().addLayout(param_layout)
        
        # Add stretch to push parameters to the top
        self.param_widget.layout().addStretch()
        
        # update button states
        self.update_button_states()
    
    def clear_parameter_widgets(self):
        """Clear all parameter widgets."""
        # rmove all widgets from the parameter layout
        while self.param_widget.layout().count():
            item = self.param_widget.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Remove widgets from sublayout
                while item.layout().count():
                    subitem = item.layout().takeAt(0)
                    if subitem.widget():
                        subitem.widget().deleteLater()
    
    def update_parameter(self, row, operation, param_name, value):
        """Update a parameter value in the pipeline."""
        if row < 0 or row >= len(self.processor.pipeline):
            return
            
        # update the parameter
        self.processor.pipeline[row]["params"][param_name] = value
    
    def process_image(self):
        """Process the image with the current pipeline."""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
            
        if not self.processor.pipeline:
            QMessageBox.warning(self, "Warning", "Pipeline is empty.")
            return
            
        # Disable process button during processing
        self.process_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        
        # Create and start the processing thread
        self.processing_thread = ProcessingThread(self.processor, self.original_image)
        self.processing_thread.processingFinished.connect(self.on_processing_finished)
        self.processing_thread.start()
    
    def on_processing_finished(self, result):
        """Handle completion of image processing."""
        self.processed_image = result
        
        # Display the processed image
        self.display_image(self.processed_image, self.processed_label)
        
        # Re-enable process button
        self.process_btn.setEnabled(True)
        
        # Enable save and full screen buttons
        self.save_image_btn.setEnabled(True)
        self.save_now_btn.setEnabled(True)
        self.full_screen_btn.setEnabled(True)
        
        # Set status
        self.status_label.setText("Processing complete")
    
    def display_image(self, cv_img, label):
        """Display an OpenCV image on a QLabel."""
        if cv_img is None:
            return
            
        # convert to QImage
        height, width = cv_img.shape
        bytes_per_line = width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # convert to QPixmap
        pixmap = QPixmap.fromImage(q_img)
        
        # sale to fit the label
        label_size = label.size()
        scaled_pixmap = pixmap.scaled(
            label_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # set the pixmap
        label.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DentalXRayEnhancerGUI()
    sys.exit(app.exec_())
