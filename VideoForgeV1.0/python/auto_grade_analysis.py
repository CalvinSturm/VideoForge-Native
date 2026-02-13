"""
VideoForge Auto Color Grading - Media Analysis Module

This module provides deterministic, explainable analysis of media for
automatic color grading suggestions. All analysis is performed on a single
representative frame using conservative algorithms.

Design Principles:
- Deterministic: Same input always produces same analysis
- Conservative: Avoid aggressive corrections
- Explainable: All outputs include confidence scores
- Skin-aware: Protect skin tones from overcorrection

Author: VideoForge Team
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

class AutoGradeConfig:
    """Configuration for auto-grading analysis"""
    
    # Analysis resolution (lower = faster, but less accurate)
    ANALYSIS_HEIGHT = 360
    
    # Exposure parameters
    TARGET_MEDIAN_Y = 128  # Target luminance median (0-255)
    EXPOSURE_TOLERANCE = 10  # Don't correct if within this range of target
    MAX_EXPOSURE_STOPS = 0.5  # Maximum ±stops adjustment
    
    # White balance parameters
    MAX_WB_CORRECTION = 0.15  # Maximum ±15% per channel
    SKIN_WB_REDUCTION = 0.5  # Reduce WB correction by 50% when skin detected
    
    # Contrast parameters
    MAX_CONTRAST_BOOST = 0.35  # Maximum S-curve strength
    
    # Saturation parameters
    MAX_SATURATION_BOOST = 0.15  # Maximum +15% saturation
    
    # Skin detection (YCrCb range)
    SKIN_Y_MIN, SKIN_Y_MAX = 0, 255
    SKIN_CR_MIN, SKIN_CR_MAX = 133, 173
    SKIN_CB_MIN, SKIN_CB_MAX = 77, 127
    SKIN_THRESHOLD = 0.05  # Minimum ratio to consider "has skin"
    
    # Clipping thresholds
    SHADOW_CLIP_THRESHOLD = 8  # Pixels below this are clipped shadows
    HIGHLIGHT_CLIP_THRESHOLD = 247  # Pixels above this are clipped highlights
    MAX_CLIP_BEFORE_SKIP = 0.10  # Skip exposure correction if >10% clipped
    
    # Log footage detection
    LOG_MID_CONCENTRATION = 0.85  # Histogram midtone concentration threshold
    LOG_DYNAMIC_RANGE = 0.4  # Maximum dynamic range for log detection


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def downsample_for_analysis(frame: np.ndarray, target_height: int = AutoGradeConfig.ANALYSIS_HEIGHT) -> np.ndarray:
    """
    Downsample frame for faster analysis while preserving color distribution.
    Uses INTER_AREA for high-quality downsampling.
    """
    h, w = frame.shape[:2]
    if h <= target_height:
        return frame
    
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]"""
    return max(min_val, min(max_val, value))


# =============================================================================
# STAGE 1: TECHNICAL MEDIA ANALYSIS
# =============================================================================

def analyze_histogram(frame: np.ndarray) -> Dict[str, Any]:
    """
    Analyze luminance histogram for exposure characteristics.
    
    Returns:
        Dictionary with:
        - histogram: 256-bin luminance histogram (normalized)
        - median_y: Median luminance value (0-255)
        - shadow_clip: Ratio of clipped shadow pixels (0-1)
        - highlight_clip: Ratio of clipped highlight pixels (0-1)
        - exposure_bias: Exposure bias in normalized units (-1 to +1)
        - dynamic_range: Dynamic range as ratio (0-1)
        - confidence: Analysis confidence (0-1)
    """
    # Convert to YCrCb and extract Y channel
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].flatten()
    
    # Calculate histogram
    hist, _ = np.histogram(y_channel, bins=256, range=(0, 256))
    hist_normalized = hist / hist.sum()
    
    # Calculate percentiles
    p1 = np.percentile(y_channel, 1)
    p50 = np.percentile(y_channel, 50)  # Median
    p99 = np.percentile(y_channel, 99)
    
    # Calculate clipping ratios
    shadow_clip = np.sum(y_channel < AutoGradeConfig.SHADOW_CLIP_THRESHOLD) / len(y_channel)
    highlight_clip = np.sum(y_channel > AutoGradeConfig.HIGHLIGHT_CLIP_THRESHOLD) / len(y_channel)
    
    # Calculate exposure bias (-1 to +1, negative = underexposed)
    exposure_bias = (p50 - AutoGradeConfig.TARGET_MEDIAN_Y) / 128.0
    
    # Calculate dynamic range (0 = very compressed, 1 = full range)
    dynamic_range = (p99 - p1) / 255.0
    
    # Confidence: lower if heavily clipped or extremely compressed
    confidence = 1.0
    if shadow_clip > 0.05 or highlight_clip > 0.05:
        confidence *= 0.7
    if dynamic_range < 0.3:
        confidence *= 0.8
    
    return {
        "histogram": hist_normalized.tolist(),
        "median_y": float(p50),
        "shadow_clip": float(shadow_clip),
        "highlight_clip": float(highlight_clip),
        "exposure_bias": float(clamp(exposure_bias, -1.0, 1.0)),
        "dynamic_range": float(dynamic_range),
        "confidence": float(confidence)
    }


def analyze_white_balance(frame: np.ndarray) -> Dict[str, Any]:
    """
    Analyze white balance using modified grey-world algorithm.
    
    Excludes very dark and very bright pixels for more accurate estimation.
    
    Returns:
        Dictionary with:
        - temp_bias: Temperature bias (-1=cool, +1=warm)
        - tint_bias: Tint bias (-1=magenta, +1=green)
        - r_avg, g_avg, b_avg: Channel averages
        - confidence: Analysis confidence (0-1)
    """
    # Convert to float for precision
    frame_f = frame.astype(np.float32)
    
    # Create mask excluding very dark and very bright pixels
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    valid_mask = (gray > 20) & (gray < 235)
    
    if np.sum(valid_mask) < 1000:
        # Not enough valid pixels, use full frame
        valid_mask = np.ones(gray.shape, dtype=bool)
    
    # Calculate weighted channel averages
    b_avg = np.mean(frame_f[:, :, 0][valid_mask])
    g_avg = np.mean(frame_f[:, :, 1][valid_mask])
    r_avg = np.mean(frame_f[:, :, 2][valid_mask])
    
    # Calculate bias (deviation from neutral gray)
    # Positive temp_bias = warm (more red), negative = cool (more blue)
    temp_bias = (r_avg - b_avg) / 255.0
    
    # Positive tint_bias = green cast, negative = magenta cast
    tint_bias = (g_avg - (r_avg + b_avg) / 2.0) / 255.0
    
    # Confidence: lower if very extreme bias (might be intentional)
    confidence = 1.0
    if abs(temp_bias) > 0.3 or abs(tint_bias) > 0.2:
        confidence *= 0.6  # Extreme bias might be intentional
    
    return {
        "temp_bias": float(clamp(temp_bias, -1.0, 1.0)),
        "tint_bias": float(clamp(tint_bias, -1.0, 1.0)),
        "r_avg": float(r_avg),
        "g_avg": float(g_avg),
        "b_avg": float(b_avg),
        "confidence": float(confidence)
    }


def analyze_noise(frame: np.ndarray) -> Dict[str, Any]:
    """
    Estimate noise level using Laplacian variance method.
    
    Also analyzes shadow regions specifically as noise is more visible there.
    
    Returns:
        Dictionary with:
        - noise_level: Overall noise estimate (0=clean, 1=very noisy)
        - shadow_noise: Noise in shadow regions specifically
        - confidence: Analysis confidence (0-1)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Overall Laplacian variance (higher = more edges OR more noise)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize to 0-1 range (empirical scaling)
    # Values above 1500 are considered very noisy
    noise_level = clamp(variance / 1500.0, 0.0, 1.0)
    
    # Analyze shadow regions specifically
    shadow_mask = gray < 50
    if np.sum(shadow_mask) > 500:
        shadow_region = gray[shadow_mask]
        shadow_variance = np.var(shadow_region)
        shadow_noise = clamp(shadow_variance / 100.0, 0.0, 1.0)
    else:
        shadow_noise = noise_level
    
    return {
        "noise_level": float(noise_level),
        "shadow_noise": float(shadow_noise),
        "confidence": 0.85  # Noise estimation is inherently imprecise
    }


def detect_skin_tones(frame: np.ndarray) -> Dict[str, Any]:
    """
    Detect presence of skin tones using YCrCb color space thresholding.
    
    Returns:
        Dictionary with:
        - has_skin: Boolean indicating significant skin presence
        - skin_ratio: Ratio of skin pixels (0-1)
        - is_face_dominant: True if skin covers >15% of frame
        - confidence: Detection confidence (0-1)
    """
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Create skin mask using empirical ranges
    skin_mask = cv2.inRange(
        ycrcb,
        (AutoGradeConfig.SKIN_Y_MIN, AutoGradeConfig.SKIN_CR_MIN, AutoGradeConfig.SKIN_CB_MIN),
        (AutoGradeConfig.SKIN_Y_MAX, AutoGradeConfig.SKIN_CR_MAX, AutoGradeConfig.SKIN_CB_MAX)
    )
    
    # Calculate ratio
    skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size
    
    has_skin = skin_ratio > AutoGradeConfig.SKIN_THRESHOLD
    is_face_dominant = skin_ratio > 0.15
    
    # Confidence based on ratio (very high or very low ratios are more confident)
    if skin_ratio < 0.02 or skin_ratio > 0.3:
        confidence = 0.9
    else:
        confidence = 0.75  # Ambiguous range
    
    return {
        "has_skin": has_skin,
        "skin_ratio": float(skin_ratio),
        "is_face_dominant": is_face_dominant,
        "confidence": float(confidence)
    }


# =============================================================================
# STAGE 2: SCENE CLASSIFICATION
# =============================================================================

def detect_log_footage(histogram_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect if footage is encoded in log gamma.
    
    Log footage has compressed histogram concentrated in midtones.
    
    Returns:
        Dictionary with:
        - is_log: Boolean indicating log footage
        - log_confidence: Confidence of log detection (0-1)
    """
    histogram = np.array(histogram_result["histogram"])
    dynamic_range = histogram_result["dynamic_range"]
    
    # Calculate midtone concentration (bins 64-192)
    mid_concentration = np.sum(histogram[64:192])
    
    is_log = (mid_concentration > AutoGradeConfig.LOG_MID_CONCENTRATION and 
              dynamic_range < AutoGradeConfig.LOG_DYNAMIC_RANGE)
    
    # Confidence calculation
    log_confidence = min(mid_concentration, 1.0 - dynamic_range) if is_log else 0.0
    
    return {
        "is_log": is_log,
        "log_confidence": float(log_confidence)
    }


def classify_scene(histogram_result: Dict[str, Any], 
                   wb_result: Dict[str, Any],
                   skin_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify scene characteristics.
    
    Returns:
        Dictionary with:
        - is_outdoor: Estimated outdoor scene (True/False/None if uncertain)
        - is_day: Estimated daylight scene (True/False/None if uncertain)
        - is_low_key: Low-key (dark) scene
        - is_high_key: High-key (bright) scene
        - log_detection: Log footage detection results
        - overall_confidence: Aggregate confidence
    """
    median_y = histogram_result["median_y"]
    temp_bias = wb_result["temp_bias"]
    
    # Detect log footage
    log_detection = detect_log_footage(histogram_result)
    
    # Key analysis based on median luminance
    is_low_key = median_y < 80
    is_high_key = median_y > 180
    
    # Day/Night estimation (daylight tends to be brighter with warmer bias)
    if median_y > 120 and temp_bias > -0.1:
        is_day = True
        day_confidence = 0.7
    elif median_y < 60:
        is_day = False
        day_confidence = 0.75
    else:
        is_day = None
        day_confidence = 0.5
    
    # Indoor/Outdoor is harder to detect from color alone
    # Warmer light often indicates indoor (tungsten), cooler might be shade
    is_outdoor = None  # Leave uncertain by default
    outdoor_confidence = 0.5
    
    # Aggregate confidence
    overall_confidence = min(
        histogram_result["confidence"],
        wb_result["confidence"],
        skin_result["confidence"]
    )
    
    # Reduce confidence for log footage (need special handling)
    if log_detection["is_log"]:
        overall_confidence *= 0.8
    
    return {
        "is_outdoor": is_outdoor,
        "is_day": is_day,
        "is_low_key": is_low_key,
        "is_high_key": is_high_key,
        "is_face_dominant": skin_result["is_face_dominant"],
        "log_detection": log_detection,
        "overall_confidence": float(overall_confidence)
    }


# =============================================================================
# STAGE 3: CORRECTION CALCULATION
# =============================================================================

def calculate_corrections(
    histogram_result: Dict[str, Any],
    wb_result: Dict[str, Any],
    skin_result: Dict[str, Any],
    scene_result: Dict[str, Any],
    conservative_mode: bool = False
) -> Dict[str, Any]:
    """
    Calculate conservative color corrections based on analysis.
    
    All corrections are clamped and scaled by confidence.
    
    Args:
        conservative_mode: If True, reduce all corrections by 40%
    
    Returns:
        Dictionary with:
        - exposure: Recommended exposure adjustment (stops)
        - temperature: Recommended temperature adjustment (-1 to +1)
        - tint: Recommended tint adjustment (-1 to +1)
        - contrast: Recommended contrast boost (0 to 1)
        - saturation: Recommended saturation boost (0 to 1)
        - applied_corrections: Human-readable list of what was applied
    """
    overall_confidence = scene_result["overall_confidence"]
    conservative_factor = 0.6 if conservative_mode else 1.0
    
    applied_corrections = []
    
    # --- EXPOSURE ---
    exposure_correction = 0.0
    exposure_bias = histogram_result["exposure_bias"]
    
    # Skip if heavily clipped
    total_clip = histogram_result["shadow_clip"] + histogram_result["highlight_clip"]
    if total_clip < AutoGradeConfig.MAX_CLIP_BEFORE_SKIP:
        # Only correct if bias is significant
        if abs(exposure_bias) * 128 > AutoGradeConfig.EXPOSURE_TOLERANCE:
            # Negative bias = underexposed, need positive correction
            raw_correction = -exposure_bias * AutoGradeConfig.MAX_EXPOSURE_STOPS
            exposure_correction = clamp(raw_correction, -0.5, 0.5)
            exposure_correction *= overall_confidence * conservative_factor
            
            if abs(exposure_correction) > 0.05:
                direction = "increased" if exposure_correction > 0 else "decreased"
                applied_corrections.append(f"Exposure {direction} by {abs(exposure_correction):.2f} stops")
    else:
        applied_corrections.append("Exposure skipped (clipping detected)")
    
    # --- WHITE BALANCE ---
    temp_correction = 0.0
    tint_correction = 0.0
    
    temp_bias = wb_result["temp_bias"]
    tint_bias = wb_result["tint_bias"]
    
    # Skin protection
    skin_protection = AutoGradeConfig.SKIN_WB_REDUCTION if skin_result["has_skin"] else 1.0
    
    # Temperature correction (counter the bias)
    if abs(temp_bias) > 0.05:
        max_correction = AutoGradeConfig.MAX_WB_CORRECTION * skin_protection
        temp_correction = clamp(-temp_bias * 0.5, -max_correction, max_correction)
        temp_correction *= overall_confidence * conservative_factor
        
        if abs(temp_correction) > 0.02:
            direction = "warmer" if temp_correction > 0 else "cooler"
            applied_corrections.append(f"White balance shifted {direction}")
    
    # Tint correction
    if abs(tint_bias) > 0.05:
        max_correction = AutoGradeConfig.MAX_WB_CORRECTION * skin_protection * 0.7
        tint_correction = clamp(-tint_bias * 0.5, -max_correction, max_correction)
        tint_correction *= overall_confidence * conservative_factor
    
    # --- CONTRAST ---
    contrast_boost = 0.0
    dynamic_range = histogram_result["dynamic_range"]
    is_log = scene_result["log_detection"]["is_log"]
    
    if is_log:
        # Log footage needs more contrast expansion
        contrast_boost = 0.3 * overall_confidence * conservative_factor
        applied_corrections.append("Log footage contrast expansion applied")
    elif dynamic_range < 0.5:
        # Low contrast footage
        contrast_boost = (0.5 - dynamic_range) * 0.4 * overall_confidence * conservative_factor
        if contrast_boost > 0.05:
            applied_corrections.append("Contrast enhanced (flat footage detected)")
    
    contrast_boost = clamp(contrast_boost, 0.0, AutoGradeConfig.MAX_CONTRAST_BOOST)
    
    # --- SATURATION ---
    saturation_boost = 0.0
    # Only boost if undersaturated (we detect this by checking color variance)
    # For now, use a conservative default for low-saturation footage
    target_sat = 1.05 if skin_result["has_skin"] else 1.10
    
    # Simple heuristic: log footage often needs saturation boost
    if is_log:
        saturation_boost = 0.08 * overall_confidence * conservative_factor
    
    saturation_boost = clamp(saturation_boost, 0.0, AutoGradeConfig.MAX_SATURATION_BOOST)
    
    if not applied_corrections:
        applied_corrections.append("No significant corrections needed")
    
    return {
        "exposure": float(exposure_correction),
        "temperature": float(temp_correction),
        "tint": float(tint_correction),
        "contrast": float(contrast_boost),
        "saturation": float(saturation_boost),
        "applied_corrections": applied_corrections,
        "overall_confidence": float(overall_confidence)
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_frame_for_auto_grade(
    frame: np.ndarray,
    protect_skin: bool = True,
    conservative_mode: bool = False
) -> Dict[str, Any]:
    """
    Complete auto-grade analysis of a single frame.
    
    This is the main entry point for auto-grading analysis.
    
    Args:
        frame: BGR image as numpy array
        protect_skin: Enable skin tone protection
        conservative_mode: Use more conservative corrections
    
    Returns:
        Dictionary with:
        - analysis: All analysis results
        - corrections: Recommended corrections
        - confidence: Overall confidence score
        - summary: Human-readable summary
    """
    # Downsample for faster analysis
    analysis_frame = downsample_for_analysis(frame)
    
    # Stage 1: Technical Analysis
    histogram_result = analyze_histogram(analysis_frame)
    wb_result = analyze_white_balance(analysis_frame)
    noise_result = analyze_noise(analysis_frame)
    skin_result = detect_skin_tones(analysis_frame) if protect_skin else {
        "has_skin": False, "skin_ratio": 0.0, "is_face_dominant": False, "confidence": 1.0
    }
    
    # Stage 2: Scene Classification
    scene_result = classify_scene(histogram_result, wb_result, skin_result)
    
    # Stage 3: Calculate Corrections
    corrections = calculate_corrections(
        histogram_result, wb_result, skin_result, scene_result, conservative_mode
    )
    
    # Build summary
    summary_parts = []
    if scene_result["is_low_key"]:
        summary_parts.append("Low-key scene")
    elif scene_result["is_high_key"]:
        summary_parts.append("High-key scene")
    if scene_result["log_detection"]["is_log"]:
        summary_parts.append("Log footage detected")
    if skin_result["has_skin"]:
        summary_parts.append("Skin tones protected")
    if noise_result["noise_level"] > 0.5:
        summary_parts.append("High noise detected")
    
    summary = "; ".join(summary_parts) if summary_parts else "Standard footage"
    
    return {
        "analysis": {
            "histogram": histogram_result,
            "white_balance": wb_result,
            "noise": noise_result,
            "skin": skin_result,
            "scene": scene_result
        },
        "corrections": corrections,
        "confidence": corrections["overall_confidence"],
        "summary": summary
    }


def convert_corrections_to_edit_config(corrections: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert auto-grade corrections to EditConfig color format.
    
    Maps our internal correction values to the existing ColorSettings format:
    - brightness: -1.0 to 1.0
    - contrast: -1.0 to 1.0
    - saturation: -1.0 to 1.0
    - gamma: 0.1 to 10.0
    
    Note: Temperature/tint are approximated via RGB adjustments in brightness.
    A proper implementation would require adding temperature/tint to ColorSettings.
    """
    # Convert exposure stops to brightness (-1 to 1)
    # 0.5 stops ≈ 0.15 brightness adjustment
    brightness = corrections["exposure"] * 0.3
    
    # Temperature approximation via gamma (warm = lower gamma on blue)
    # This is a simplification - proper WB needs per-channel adjustment
    gamma = 1.0
    
    # Contrast: our 0-0.35 maps to 0-0.35 in contrast setting
    contrast = corrections["contrast"]
    
    # Saturation: our 0-0.15 maps directly
    saturation = corrections["saturation"]
    
    return {
        "brightness": float(clamp(brightness, -1.0, 1.0)),
        "contrast": float(clamp(contrast, -1.0, 1.0)),
        "saturation": float(clamp(saturation, -1.0, 1.0)),
        "gamma": float(clamp(gamma, 0.1, 10.0))
    }


# =============================================================================
# CLI INTERFACE FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python auto_grade_analysis.py <image_path> [--conservative] [--no-skin-protection]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    conservative = "--conservative" in sys.argv
    protect_skin = "--no-skin-protection" not in sys.argv
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)
    
    # Run analysis
    result = analyze_frame_for_auto_grade(frame, protect_skin, conservative)
    
    # Print results
    print(json.dumps(result, indent=2))
    
    # Print edit config conversion
    edit_config = convert_corrections_to_edit_config(result["corrections"])
    print("\n--- Edit Config ---")
    print(json.dumps(edit_config, indent=2))
