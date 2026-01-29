use std::ffi::c_void;

#[allow(non_camel_case_types)]
pub type nvjpegHandle_t = *mut nvjpegHandle;
#[allow(non_camel_case_types)]
pub type nvjpegEncoderState_t = *mut nvjpegEncoderState;
#[allow(non_camel_case_types)]
pub type nvjpegEncoderParams_t = *mut nvjpegEncoderParams;
#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut c_void;

#[repr(C)]
pub struct nvjpegHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct nvjpegEncoderState {
    _private: [u8; 0],
}

#[repr(C)]
pub struct nvjpegEncoderParams {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct nvjpegImage_t {
    pub channel: [*mut u8; 4],
    pub pitch: [usize; 4],
}

#[allow(non_camel_case_types)]
pub type nvjpegStatus_t = i32;
pub const NVJPEG_STATUS_SUCCESS: nvjpegStatus_t = 0;

#[allow(non_camel_case_types)]
pub type nvjpegInputFormat_t = i32;
pub const NVJPEG_INPUT_RGBI: nvjpegInputFormat_t = 5;
