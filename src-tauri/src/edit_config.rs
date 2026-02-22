use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

// -----------------------------------------------------------------------------
// EditConfig - Mirrors frontend EditState
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditConfig {
    /// Trim start time in seconds
    pub trim_start: f64,
    /// Trim end time in seconds (0.0 means no trim end)
    pub trim_end: f64,
    /// Crop region as normalized coordinates (0.0-1.0)
    pub crop: Option<CropRegion>,
    /// Rotation in degrees (only 0, 90, 180, 270 allowed)
    pub rotation: Rotation,
    /// Horizontal flip
    pub flip_h: bool,
    /// Vertical flip
    pub flip_v: bool,
    /// Target Frame Rate (0 = Original)
    pub fps: u32,
    /// Color grading settings
    pub color: ColorSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CropRegion {
    /// X position as normalized coordinate (0.0-1.0)
    pub x: f64,
    /// Y position as normalized coordinate (0.0-1.0)
    pub y: f64,
    /// Width as normalized coordinate (0.0-1.0)
    pub width: f64,
    /// Height as normalized coordinate (0.0-1.0)
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorSettings {
    /// Brightness adjustment: -1.0 to 1.0 (0 = no change)
    pub brightness: f64,
    /// Contrast adjustment: -1.0 to 1.0 (0 = no change)
    pub contrast: f64,
    /// Saturation adjustment: -1.0 to 1.0 (0 = no change)
    pub saturation: f64,
    /// Gamma adjustment: 0.1 to 10.0 (1.0 = no change)
    pub gamma: f64,
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            gamma: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    Degrees0 = 0,
    Degrees90 = 90,
    Degrees180 = 180,
    Degrees270 = 270,
}

// Custom Serde implementation to handle integer values from JSON
impl<'de> Deserialize<'de> for Rotation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = u16::deserialize(deserializer)?;
        match v {
            0 => Ok(Rotation::Degrees0),
            90 => Ok(Rotation::Degrees90),
            180 => Ok(Rotation::Degrees180),
            270 => Ok(Rotation::Degrees270),
            _ => Err(serde::de::Error::custom(format!(
                "Invalid rotation value: {}",
                v
            ))),
        }
    }
}

impl Serialize for Rotation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u16(*self as u16)
    }
}

impl Default for EditConfig {
    fn default() -> Self {
        Self {
            trim_start: 0.0,
            trim_end: 0.0,
            crop: None,
            rotation: Rotation::Degrees0,
            flip_h: false,
            flip_v: false,
            fps: 0,
            color: ColorSettings::default(),
        }
    }
}

impl fmt::Display for Rotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", *self as i32)
    }
}

// -----------------------------------------------------------------------------
// FFmpeg Filter Chain Builder
// -----------------------------------------------------------------------------

pub struct FilterChainBuilder {
    filters: Vec<String>,
}

impl FilterChainBuilder {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add trim filter (always first if present)
    pub fn add_trim(&mut self, _start: f64, _end: f64) -> &mut Self {
        // Trim is currently handled via -ss/-t input args in the pipeline for efficiency,
        // but we keep this stub if we need complex filter-based trimming later.
        self
    }

    /// Add crop filter
    pub fn add_crop(
        &mut self,
        crop: &CropRegion,
        video_width: usize,
        video_height: usize,
    ) -> &mut Self {
        let crop_x = (crop.x * video_width as f64).round() as usize;
        let crop_y = (crop.y * video_height as f64).round() as usize;
        let crop_width = (crop.width * video_width as f64).round() as usize;
        let crop_height = (crop.height * video_height as f64).round() as usize;

        // Ensure valid ffmpeg crop syntax
        let filter = format!("crop={}:{}:{}:{}", crop_width, crop_height, crop_x, crop_y);
        self.filters.push(filter);
        self
    }

    /// Add rotation and flip transformations
    pub fn add_transform(&mut self, rotation: Rotation, flip_h: bool, flip_v: bool) -> &mut Self {
        let mut transform_parts = Vec::new();

        match rotation {
            Rotation::Degrees90 => transform_parts.push("transpose=1".to_string()),
            Rotation::Degrees180 => transform_parts.push("transpose=1,transpose=1".to_string()),
            Rotation::Degrees270 => transform_parts.push("transpose=2".to_string()),
            Rotation::Degrees0 => {}
        }

        if flip_h {
            transform_parts.push("hflip".to_string());
        }
        if flip_v {
            transform_parts.push("vflip".to_string());
        }

        if !transform_parts.is_empty() {
            self.filters.push(transform_parts.join(","));
        }
        self
    }

    /// Add color correction filter (FFmpeg eq filter)
    pub fn add_color(&mut self, color: &ColorSettings) -> &mut Self {
        // Check if any color adjustment is needed
        let has_adjustment = color.brightness.abs() > 0.001
            || color.contrast.abs() > 0.001
            || color.saturation.abs() > 0.001
            || (color.gamma - 1.0).abs() > 0.001;

        if !has_adjustment {
            return self;
        }

        // FFmpeg eq filter mapping:
        // brightness: -1.0 to 1.0 (maps directly)
        // contrast: eq expects 0.0-2.0 range, so we map -1.0,1.0 -> 0.0,2.0
        // saturation: eq expects 0.0-3.0 range, so we map -1.0,1.0 -> 0.0,2.0
        // gamma: 0.1 to 10.0 (maps directly)
        let eq_brightness = color.brightness;
        let eq_contrast = 1.0 + color.contrast; // -1 to 1 becomes 0 to 2
        let eq_saturation = 1.0 + color.saturation; // -1 to 1 becomes 0 to 2
        let eq_gamma = color.gamma.clamp(0.1, 10.0);

        let filter = format!(
            "eq=brightness={:.3}:contrast={:.3}:saturation={:.3}:gamma={:.3}",
            eq_brightness, eq_contrast, eq_saturation, eq_gamma
        );
        self.filters.push(filter);
        self
    }

    /// Build the complete filter chain string
    pub fn build(self) -> String {
        if self.filters.is_empty() {
            return String::new();
        }
        self.filters.join(",")
    }
}

impl Default for FilterChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------------
// Helpers exposed to lib.rs / pipeline
// -----------------------------------------------------------------------------

/// Wrapper to build filters string for FFmpeg
pub fn build_ffmpeg_filters(config: &EditConfig, input_w: usize, input_h: usize) -> String {
    let mut builder = FilterChainBuilder::new();

    if let Some(crop) = &config.crop {
        builder.add_crop(crop, input_w, input_h);
    }

    builder.add_transform(config.rotation, config.flip_h, config.flip_v);

    // Add color grading filter
    builder.add_color(&config.color);

    builder.build()
}

/// Calculate the final dimensions after crop and rotation.
/// Required for initializing the Shared Memory buffer.
pub fn calculate_output_dimensions(
    config: &EditConfig,
    input_w: usize,
    input_h: usize,
) -> (usize, usize) {
    // 1. Calculate Cropped Dimensions
    let (mut w, mut h) = if let Some(crop) = &config.crop {
        (
            (input_w as f64 * crop.width).round() as usize,
            (input_h as f64 * crop.height).round() as usize,
        )
    } else {
        (input_w, input_h)
    };

    // Ensure even dimensions (Standard video requirement)
    if w % 2 != 0 {
        w -= 1;
    }
    if h % 2 != 0 {
        h -= 1;
    }

    // 2. Swap dimensions if rotated 90 or 270 degrees
    match config.rotation {
        Rotation::Degrees90 | Rotation::Degrees270 => std::mem::swap(&mut w, &mut h),
        _ => {}
    }

    (w, h)
}

impl EditConfig {
    pub fn validate(&self) -> Result<()> {
        if self.trim_start < 0.0 {
            return Err(anyhow::anyhow!("Trim start negative"));
        }
        if self.trim_end < 0.0 {
            return Err(anyhow::anyhow!("Trim end negative"));
        }
        if self.trim_start > 0.0 && self.trim_end > 0.0 && self.trim_start >= self.trim_end {
            return Err(anyhow::anyhow!("Trim start must be < end"));
        }
        if let Some(crop) = &self.crop {
            if crop.x < 0.0
                || crop.x > 1.0
                || crop.y < 0.0
                || crop.y > 1.0
                || crop.width <= 0.0
                || crop.width > 1.0
                || crop.height <= 0.0
                || crop.height > 1.0
            {
                return Err(anyhow::anyhow!("Invalid crop normalized coords"));
            }
        }
        Ok(())
    }
}
