//! Windows named-event helpers for optional SHM sync hints.
//!
//! Events are advisory only. SHM slot state remains authoritative.

#[cfg(windows)]
mod imp {
    use std::ffi::OsStr;
    use std::iter;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;

    type Handle = *mut core::ffi::c_void;

    const SYNCHRONIZE: u32 = 0x0010_0000;
    const EVENT_MODIFY_STATE: u32 = 0x0002;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateEventW(
            lp_event_attributes: *mut core::ffi::c_void,
            b_manual_reset: i32,
            b_initial_state: i32,
            lp_name: *const u16,
        ) -> Handle;
        fn SetEvent(h_event: Handle) -> i32;
        fn CloseHandle(h_object: Handle) -> i32;
        fn OpenEventW(dw_desired_access: u32, b_inherit_handle: i32, lp_name: *const u16)
            -> Handle;
    }

    fn to_wide(name: &str) -> Vec<u16> {
        OsStr::new(name)
            .encode_wide()
            .chain(iter::once(0))
            .collect()
    }

    #[derive(Debug)]
    pub struct NamedEvent {
        handle: Handle,
        name: String,
    }

    // Win32 HANDLEs are process-wide kernel objects that can be shared across threads.
    unsafe impl Send for NamedEvent {}
    unsafe impl Sync for NamedEvent {}

    impl NamedEvent {
        pub fn create(name: &str) -> Result<Self, String> {
            let wide = to_wide(name);
            // Auto-reset event (manual_reset=false), initially non-signaled.
            let handle = unsafe { CreateEventW(ptr::null_mut(), 0, 0, wide.as_ptr()) };
            if handle.is_null() {
                return Err(format!("CreateEventW failed for '{name}'"));
            }
            Ok(Self {
                handle,
                name: name.to_string(),
            })
        }

        pub fn signal(&self) -> Result<(), String> {
            let ok = unsafe { SetEvent(self.handle) };
            if ok == 0 {
                return Err(format!("SetEvent failed for '{}'", self.name));
            }
            Ok(())
        }

        pub fn name(&self) -> &str {
            &self.name
        }
    }

    impl Drop for NamedEvent {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                let _ = unsafe { CloseHandle(self.handle) };
                self.handle = ptr::null_mut();
            }
        }
    }

    pub fn create_named_event(name: &str) -> Result<NamedEvent, String> {
        NamedEvent::create(name)
    }

    #[allow(dead_code)]
    pub fn can_open_named_event(name: &str) -> bool {
        let wide = to_wide(name);
        let handle = unsafe { OpenEventW(SYNCHRONIZE | EVENT_MODIFY_STATE, 0, wide.as_ptr()) };
        if handle.is_null() {
            return false;
        }
        let _ = unsafe { CloseHandle(handle) };
        true
    }
}

#[cfg(not(windows))]
mod imp {
    #[derive(Debug)]
    pub struct NamedEvent {
        name: String,
    }

    impl NamedEvent {
        pub fn name(&self) -> &str {
            &self.name
        }

        #[allow(clippy::unnecessary_wraps)] // Signature matches Windows implementation.
        pub fn signal(&self) -> Result<(), String> {
            Ok(())
        }
    }

    #[allow(clippy::unnecessary_wraps)] // Signature matches Windows implementation.
    pub fn create_named_event(name: &str) -> Result<NamedEvent, String> {
        Ok(NamedEvent {
            name: name.to_string(),
        })
    }
}

pub use imp::{create_named_event, NamedEvent};

/// Build a deterministic Win32-safe event name for a given job and role.
///
/// Naming scheme:
/// - `vf_shm_{sanitized_job_id}_{suffix}`
/// - `sanitized_job_id` keeps only ASCII alnum, converting others to `_`
/// - Length is capped to keep names practical for Win32 object namespaces
pub fn format_event_name(job_id: &str, suffix: &str) -> String {
    let safe_job: String = job_id
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    let mut name = format!("vf_shm_{}_{}", safe_job, suffix);
    if name.len() > 120 {
        name.truncate(120);
    }
    name
}

#[cfg(test)]
mod tests {
    use super::format_event_name;

    #[test]
    fn event_name_is_deterministic_and_safe() {
        let a = format_event_name("job-123/abc", "in_ready");
        let b = format_event_name("job-123/abc", "in_ready");
        assert_eq!(a, b);
        assert!(a.starts_with("vf_shm_"));
        assert!(a.ends_with("in_ready"));
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    }

    #[test]
    fn event_name_is_bounded() {
        let long_id = "x".repeat(256);
        let name = format_event_name(&long_id, "out_ready");
        assert!(name.len() <= 120);
    }
}
