//! Spatial Map Publisher — serializes and publishes spatial routing maps over Zenoh.
//!
//! Binary wire format (little-endian):
//! ```text
//! [0..4)  u32   width
//! [4..8)  u32   height
//! [8..)   u8[]  mask  (width × height, row-major)
//! ```
//!
//! Classification values: 0 = Flat, 1 = Texture, 2 = Edge

use zenoh::pubsub::Publisher;

/// Canonical Zenoh topic for spatial routing maps.
pub const SPATIAL_MAP_TOPIC: &str = "videoforge/research/spatial_map";

const HEADER_SIZE: usize = 8; // 2 × u32

/// Reusable publisher that avoids per-frame heap allocations.
///
/// # Example
/// ```no_run
/// # use app_lib::spatial_publisher::SpatialMapPublisher;
/// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let session = zenoh::open(zenoh::Config::default()).await?;
/// let mut pub_ = SpatialMapPublisher::new(&session).await?;
///
/// let mask = vec![0u8; 1920 * 1080]; // all Flat
/// pub_.publish(1920, 1080, &mask).await?;
///
/// session.close().await?;
/// # Ok(())
/// # }
/// ```
pub struct SpatialMapPublisher {
    publisher: Publisher<'static>,
    buf: Vec<u8>,
}

impl SpatialMapPublisher {
    /// Create a publisher bound to [`SPATIAL_MAP_TOPIC`].
    pub async fn new(session: &zenoh::Session) -> Result<Self, zenoh::Error> {
        let publisher = session
            .declare_publisher(SPATIAL_MAP_TOPIC)
            .await?;
        Ok(Self {
            publisher,
            // Pre-allocate for 1080p — grows if needed, never shrinks
            buf: Vec::with_capacity(HEADER_SIZE + 1920 * 1080),
        })
    }

    /// Serialize and publish a spatial routing map.
    ///
    /// The buffer is reused across calls to avoid repeated allocation.
    pub async fn publish(
        &mut self,
        width: u32,
        height: u32,
        mask: &[u8],
    ) -> Result<(), SpatialPublishError> {
        let expected = (width as usize) * (height as usize);
        if mask.len() != expected {
            return Err(SpatialPublishError::MaskLength {
                got: mask.len(),
                expected,
            });
        }

        // Reuse buffer — clear + extend avoids dealloc when capacity suffices
        self.buf.clear();
        self.buf.reserve(HEADER_SIZE + expected);
        self.buf.extend_from_slice(&width.to_le_bytes());
        self.buf.extend_from_slice(&height.to_le_bytes());
        self.buf.extend_from_slice(mask);

        self.publisher
            .put(self.buf.as_slice())
            .await
            .map_err(SpatialPublishError::Zenoh)?;

        Ok(())
    }
}

/// One-shot publish without reusable state.
/// Prefer [`SpatialMapPublisher`] in hot loops.
pub async fn publish_spatial_map(
    session: &zenoh::Session,
    width: u32,
    height: u32,
    mask: &[u8],
) -> Result<(), SpatialPublishError> {
    let expected = (width as usize) * (height as usize);
    if mask.len() != expected {
        return Err(SpatialPublishError::MaskLength {
            got: mask.len(),
            expected,
        });
    }

    let mut buf = Vec::with_capacity(HEADER_SIZE + expected);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf.extend_from_slice(mask);

    session
        .put(SPATIAL_MAP_TOPIC, buf.as_slice())
        .await
        .map_err(SpatialPublishError::Zenoh)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SpatialPublishError {
    MaskLength { got: usize, expected: usize },
    Zenoh(zenoh::Error),
}

impl std::fmt::Display for SpatialPublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaskLength { got, expected } => {
                write!(f, "mask length {got} != expected {expected}")
            }
            Self::Zenoh(e) => write!(f, "zenoh: {e}"),
        }
    }
}

impl std::error::Error for SpatialPublishError {}
