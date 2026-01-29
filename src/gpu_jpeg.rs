use crate::AppError;
use image::RgbaImage;

pub fn convert_png_to_jpeg_gpu(rgba: &RgbaImage, quality: u8) -> Result<Vec<u8>, AppError> {
    #[cfg(target_os = "linux")]
    {
        return linux::convert_png_to_jpeg_gpu(rgba, quality);
    }
    #[cfg(target_os = "macos")]
    {
        return macos::convert_png_to_jpeg_gpu(rgba, quality);
    }
    #[cfg(target_os = "windows")]
    {
        return windows::convert_png_to_jpeg_gpu(rgba, quality);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err(AppError::Invalid(
            "jpg_gpu is supported only on Linux, macOS, or Windows".to_string(),
        ))
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::env;
    use std::path::PathBuf;
    use std::ptr;

    use libloading::Library;

    use crate::ffi::nvjpeg_bindings::{
        cudaStream_t, nvjpegEncoderParams_t, nvjpegEncoderState_t, nvjpegHandle_t, nvjpegImage_t,
        nvjpegInputFormat_t, nvjpegStatus_t, NVJPEG_INPUT_RGBI, NVJPEG_STATUS_SUCCESS,
    };
    use crate::AppError;
    use image::RgbaImage;

    type NvjpegCreateSimple = unsafe extern "C" fn(*mut nvjpegHandle_t) -> nvjpegStatus_t;
    type NvjpegDestroy = unsafe extern "C" fn(nvjpegHandle_t) -> nvjpegStatus_t;
    type NvjpegEncoderStateCreate =
        unsafe extern "C" fn(nvjpegHandle_t, *mut nvjpegEncoderState_t, cudaStream_t)
            -> nvjpegStatus_t;
    type NvjpegEncoderStateDestroy = unsafe extern "C" fn(nvjpegEncoderState_t) -> nvjpegStatus_t;
    type NvjpegEncoderParamsCreate =
        unsafe extern "C" fn(nvjpegHandle_t, *mut nvjpegEncoderParams_t, cudaStream_t)
            -> nvjpegStatus_t;
    type NvjpegEncoderParamsDestroy =
        unsafe extern "C" fn(nvjpegEncoderParams_t) -> nvjpegStatus_t;
    type NvjpegEncoderParamsSetQuality =
        unsafe extern "C" fn(nvjpegEncoderParams_t, i32, cudaStream_t) -> nvjpegStatus_t;
    type NvjpegEncodeImage = unsafe extern "C" fn(
        nvjpegHandle_t,
        nvjpegEncoderState_t,
        nvjpegEncoderParams_t,
        *const nvjpegImage_t,
        nvjpegInputFormat_t,
        i32,
        i32,
        cudaStream_t,
    ) -> nvjpegStatus_t;
    type NvjpegEncodeRetrieveBitstream = unsafe extern "C" fn(
        nvjpegHandle_t,
        nvjpegEncoderState_t,
        *mut u8,
        *mut usize,
        cudaStream_t,
    ) -> nvjpegStatus_t;

    struct NvJpegFns {
        create_simple: NvjpegCreateSimple,
        destroy: NvjpegDestroy,
        encoder_state_create: NvjpegEncoderStateCreate,
        encoder_state_destroy: NvjpegEncoderStateDestroy,
        encoder_params_create: NvjpegEncoderParamsCreate,
        encoder_params_destroy: NvjpegEncoderParamsDestroy,
        encoder_params_set_quality: NvjpegEncoderParamsSetQuality,
        encode_image: NvjpegEncodeImage,
        encode_retrieve: NvjpegEncodeRetrieveBitstream,
    }

    impl NvJpegFns {
        unsafe fn load(lib: &Library) -> Result<Self, AppError> {
            unsafe {
                Ok(Self {
                    create_simple: *lib
                        .get(b"nvjpegCreateSimple\0")
                        .map_err(|e| AppError::Invalid(format!("nvjpegCreateSimple: {}", e)))?,
                    destroy: *lib
                        .get(b"nvjpegDestroy\0")
                        .map_err(|e| AppError::Invalid(format!("nvjpegDestroy: {}", e)))?,
                    encoder_state_create: *lib
                        .get(b"nvjpegEncoderStateCreate\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderStateCreate: {}", e))
                        })?,
                    encoder_state_destroy: *lib
                        .get(b"nvjpegEncoderStateDestroy\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderStateDestroy: {}", e))
                        })?,
                    encoder_params_create: *lib
                        .get(b"nvjpegEncoderParamsCreate\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderParamsCreate: {}", e))
                        })?,
                    encoder_params_destroy: *lib
                        .get(b"nvjpegEncoderParamsDestroy\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderParamsDestroy: {}", e))
                        })?,
                    encoder_params_set_quality: *lib
                        .get(b"nvjpegEncoderParamsSetQuality\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderParamsSetQuality: {}", e))
                        })?,
                    encode_image: *lib
                        .get(b"nvjpegEncodeImage\0")
                        .map_err(|e| AppError::Invalid(format!("nvjpegEncodeImage: {}", e)))?,
                    encode_retrieve: *lib
                        .get(b"nvjpegEncodeRetrieveBitstream\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncodeRetrieveBitstream: {}", e))
                        })?,
                })
            }
        }
    }

    struct NvJpeg {
        _lib: Library,
        fns: NvJpegFns,
    }

    impl NvJpeg {
        fn load() -> Result<Self, AppError> {
            let path = nvjpeg_library_path();
            let lib = unsafe {
                Library::new(&path)
                    .map_err(|e| AppError::Invalid(format!("nvjpeg library load: {}", e)))?
            };
            let fns = unsafe { NvJpegFns::load(&lib)? };
            Ok(Self { _lib: lib, fns })
        }
    }

    struct NvJpegContext<'a> {
        fns: &'a NvJpegFns,
        handle: nvjpegHandle_t,
        state: nvjpegEncoderState_t,
        params: nvjpegEncoderParams_t,
    }

    impl<'a> NvJpegContext<'a> {
        fn new(fns: &'a NvJpegFns) -> Self {
            Self {
                fns,
                handle: ptr::null_mut(),
                state: ptr::null_mut(),
                params: ptr::null_mut(),
            }
        }
    }

    impl Drop for NvJpegContext<'_> {
        fn drop(&mut self) {
            unsafe {
                if !self.params.is_null() {
                    let _ = (self.fns.encoder_params_destroy)(self.params);
                }
                if !self.state.is_null() {
                    let _ = (self.fns.encoder_state_destroy)(self.state);
                }
                if !self.handle.is_null() {
                    let _ = (self.fns.destroy)(self.handle);
                }
            }
        }
    }

    pub(super) fn convert_png_to_jpeg_gpu(
        rgba: &RgbaImage,
        quality: u8,
    ) -> Result<Vec<u8>, AppError> {
        let nvjpeg = NvJpeg::load()?;
        let mut ctx = NvJpegContext::new(&nvjpeg.fns);
        let stream: cudaStream_t = ptr::null_mut();

        unsafe {
            check_status(
                (nvjpeg.fns.create_simple)(&mut ctx.handle),
                "create handle",
            )?;
            check_status(
                (nvjpeg.fns.encoder_state_create)(ctx.handle, &mut ctx.state, stream),
                "create encoder state",
            )?;
            check_status(
                (nvjpeg.fns.encoder_params_create)(ctx.handle, &mut ctx.params, stream),
                "create encoder params",
            )?;
            let quality = quality.clamp(1, 100) as i32;
            check_status(
                (nvjpeg.fns.encoder_params_set_quality)(ctx.params, quality, stream),
                "set quality",
            )?;
        }

        let width = rgba.width() as i32;
        let height = rgba.height() as i32;
        if width <= 0 || height <= 0 {
            return Err(AppError::Invalid("Invalid image size".to_string()));
        }

        let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
        for chunk in rgba.as_raw().chunks_exact(4) {
            rgb.push(chunk[0]);
            rgb.push(chunk[1]);
            rgb.push(chunk[2]);
        }

        let mut image = nvjpegImage_t {
            channel: [ptr::null_mut(); 4],
            pitch: [0; 4],
        };
        image.channel[0] = rgb.as_mut_ptr();
        image.pitch[0] = (rgba.width() as usize) * 3;

        unsafe {
            check_status(
                (nvjpeg.fns.encode_image)(
                    ctx.handle,
                    ctx.state,
                    ctx.params,
                    &image,
                    NVJPEG_INPUT_RGBI,
                    width,
                    height,
                    stream,
                ),
                "encode image",
            )?;
            let mut length: usize = 0;
            check_status(
                (nvjpeg.fns.encode_retrieve)(ctx.handle, ctx.state, ptr::null_mut(), &mut length, stream),
                "get bitstream size",
            )?;
            if length == 0 {
                return Err(AppError::Invalid("nvjpeg produced empty output".to_string()));
            }
            let mut out = vec![0u8; length];
            check_status(
                (nvjpeg.fns.encode_retrieve)(
                    ctx.handle,
                    ctx.state,
                    out.as_mut_ptr(),
                    &mut length,
                    stream,
                ),
                "get bitstream",
            )?;
            out.truncate(length);
            Ok(out)
        }
    }

    fn nvjpeg_library_path() -> PathBuf {
        if let Ok(value) = env::var("NVJPEG_LIB") {
            let path = PathBuf::from(value);
            if path.is_dir() {
                return path.join("libnvjpeg.so");
            }
            return path;
        }
        PathBuf::from("libnvjpeg.so")
    }

    fn check_status(status: nvjpegStatus_t, context: &str) -> Result<(), AppError> {
        if status == NVJPEG_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(AppError::Invalid(format!(
                "nvjpeg {} failed: {}",
                context, status
            )))
        }
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::sync::{Arc, Condvar, Mutex};

    use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFString, CFType, CFRetained};
    use objc2_core_media::{CMSampleBuffer, kCMTimeInvalid, kCMTimeZero, kCMVideoCodecType_JPEG};
    use objc2_core_video::{
        CVPixelBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
        CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
        kCVPixelFormatType_32BGRA, kCVReturnSuccess, CVPixelBufferCreate,
    };
    use objc2_video_toolbox::{
        kVTCompressionPropertyKey_Quality,
        kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
        VTCompressionSession, VTEncodeInfoFlags, VTSessionSetProperty,
    };

    use crate::AppError;
    use image::RgbaImage;

    struct CallbackState {
        data: Mutex<Option<Vec<u8>>>,
        status: Mutex<Option<i32>>,
        done: Condvar,
    }

    impl CallbackState {
        fn new() -> Self {
            Self {
                data: Mutex::new(None),
                status: Mutex::new(None),
                done: Condvar::new(),
            }
        }
    }

    unsafe extern "C-unwind" fn output_callback(
        output_callback_ref_con: *mut c_void,
        _source_frame_ref_con: *mut c_void,
        status: i32,
        _info_flags: VTEncodeInfoFlags,
        sample_buffer: *mut CMSampleBuffer,
    ) {
        if output_callback_ref_con.is_null() {
            return;
        }
        let state = unsafe { &*(output_callback_ref_con as *const CallbackState) };

        let mut status_guard = state.status.lock().unwrap();
        *status_guard = Some(status);
        drop(status_guard);

        if status == 0 && !sample_buffer.is_null() {
            let buffer = unsafe { &*sample_buffer };
            let data_buffer = unsafe { buffer.data_buffer() };
            if let Some(block) = data_buffer {
                let length = unsafe { block.data_length() };
                if length > 0 {
                    let mut out = vec![0u8; length];
                    let dest = NonNull::new(out.as_mut_ptr().cast::<c_void>());
                    if let Some(dest) = dest {
                        let copy_status = unsafe { block.copy_data_bytes(0, length, dest) };
                        if copy_status == 0 {
                            let mut data_guard = state.data.lock().unwrap();
                            *data_guard = Some(out);
                        } else {
                            let mut status_guard = state.status.lock().unwrap();
                            *status_guard = Some(copy_status);
                        }
                    }
                }
            }
        }

        state.done.notify_one();
    }

    pub(super) fn convert_png_to_jpeg_gpu(
        rgba: &RgbaImage,
        quality: u8,
    ) -> Result<Vec<u8>, AppError> {
        let width = rgba.width() as usize;
        let height = rgba.height() as usize;
        if width == 0 || height == 0 {
            return Err(AppError::Invalid("Invalid image size".to_string()));
        }

        let state = Arc::new(CallbackState::new());
        let state_raw = Arc::into_raw(state.clone()) as *mut c_void;

        let encoder_spec = build_encoder_spec()?;
        let encoder_spec_ref = unsafe {
            &*(encoder_spec.as_ref() as *const CFDictionary<CFString, CFType> as *const CFDictionary)
        };

        let mut session_ptr: *mut VTCompressionSession = std::ptr::null_mut();
        let session_out = NonNull::new(&mut session_ptr as *mut *mut VTCompressionSession)
            .ok_or_else(|| AppError::Invalid("Invalid compression session pointer".to_string()))?;
        let status = unsafe {
            VTCompressionSession::create(
                None,
                width as i32,
                height as i32,
                kCMVideoCodecType_JPEG,
                Some(encoder_spec_ref),
                None,
                None,
                Some(output_callback),
                state_raw,
                session_out,
            )
        };
        if status != 0 {
            unsafe {
                Arc::from_raw(state_raw as *const CallbackState);
            }
            return Err(AppError::Invalid(format!(
                "VideoToolbox session create failed: {}",
                status
            )));
        }

        let session_ptr = NonNull::new(session_ptr)
            .ok_or_else(|| AppError::Invalid("Compression session is null".to_string()))?;
        let session = unsafe { CFRetained::from_raw(session_ptr) };

        let quality_value = CFNumber::new_f32(quality.clamp(1, 100) as f32 / 100.0);
        let status = unsafe {
            VTSessionSetProperty(
                session.as_ref(),
                kVTCompressionPropertyKey_Quality,
                Some(quality_value.as_ref()),
            )
        };
        if status != 0 {
            unsafe {
                Arc::from_raw(state_raw as *const CallbackState);
            }
            return Err(AppError::Invalid(format!(
                "VideoToolbox set quality failed: {}",
                status
            )));
        }

        let pixel_buffer = create_pixel_buffer(width, height)?;
        fill_pixel_buffer(&pixel_buffer, rgba)?;

        let zero_time = unsafe { kCMTimeZero };
        let encode_status = unsafe {
            session.encode_frame(
                &pixel_buffer,
                zero_time,
                zero_time,
                None,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if encode_status != 0 {
            unsafe {
                Arc::from_raw(state_raw as *const CallbackState);
            }
            return Err(AppError::Invalid(format!(
                "VideoToolbox encode failed: {}",
                encode_status
            )));
        }

        let invalid_time = unsafe { kCMTimeInvalid };
        let complete_status = unsafe { session.complete_frames(invalid_time) };
        if complete_status != 0 {
            unsafe {
                Arc::from_raw(state_raw as *const CallbackState);
            }
            return Err(AppError::Invalid(format!(
                "VideoToolbox complete failed: {}",
                complete_status
            )));
        }

        let mut status_guard = state.status.lock().unwrap();
        while status_guard.is_none() {
            status_guard = state.done.wait(status_guard).unwrap();
        }
        let final_status = status_guard.unwrap_or(-1);
        drop(status_guard);

        unsafe {
            Arc::from_raw(state_raw as *const CallbackState);
        }

        if final_status != 0 {
            return Err(AppError::Invalid(format!(
                "VideoToolbox callback failed: {}",
                final_status
            )));
        }

        let mut data_guard = state.data.lock().unwrap();
        if let Some(data) = data_guard.take() {
            Ok(data)
        } else {
            Err(AppError::Invalid(
                "VideoToolbox returned empty output".to_string(),
            ))
        }
    }

    fn build_encoder_spec() -> Result<CFRetained<CFDictionary<CFString, CFType>>, AppError> {
        let require_hw = CFBoolean::new(true);
        let require_hw_key = unsafe { kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder };
        let keys = [require_hw_key];
        let values: [&CFType; 1] = [require_hw.as_ref()];
        Ok(CFDictionary::from_slices(&keys, &values))
    }

    fn create_pixel_buffer(
        width: usize,
        height: usize,
    ) -> Result<CFRetained<CVPixelBuffer>, AppError> {
        let mut pixel_buffer: *mut CVPixelBuffer = std::ptr::null_mut();
        let pixel_buffer_out =
            NonNull::new(&mut pixel_buffer as *mut *mut CVPixelBuffer).ok_or_else(|| {
                AppError::Invalid("Invalid pixel buffer pointer".to_string())
            })?;
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                width,
                height,
                kCVPixelFormatType_32BGRA,
                None,
                pixel_buffer_out,
            )
        };
        if status != kCVReturnSuccess {
            return Err(AppError::Invalid(format!(
                "CVPixelBufferCreate failed: {}",
                status
            )));
        }
        let pixel_buffer = NonNull::new(pixel_buffer)
            .ok_or_else(|| AppError::Invalid("Pixel buffer is null".to_string()))?;
        Ok(unsafe { CFRetained::from_raw(pixel_buffer) })
    }

    fn fill_pixel_buffer(pixel_buffer: &CVPixelBuffer, rgba: &RgbaImage) -> Result<(), AppError> {
        let lock_status = unsafe {
            CVPixelBufferLockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::empty())
        };
        if lock_status != kCVReturnSuccess {
            return Err(AppError::Invalid(format!(
                "CVPixelBufferLockBaseAddress failed: {}",
                lock_status
            )));
        }

        let base_address = CVPixelBufferGetBaseAddress(pixel_buffer);
        if base_address.is_null() {
            unsafe {
                CVPixelBufferUnlockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::empty());
            }
            return Err(AppError::Invalid(
                "CVPixelBuffer base address is null".to_string(),
            ));
        }

        let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
        let width = rgba.width() as usize;
        let height = rgba.height() as usize;

        unsafe {
            let src = rgba.as_raw();
            for y in 0..height {
                let dst_row = (base_address as *mut u8).add(y * bytes_per_row);
                let src_row = &src[y * width * 4..(y + 1) * width * 4];
                for x in 0..width {
                    let src_idx = x * 4;
                    let dst_idx = x * 4;
                    *dst_row.add(dst_idx) = src_row[src_idx + 2];
                    *dst_row.add(dst_idx + 1) = src_row[src_idx + 1];
                    *dst_row.add(dst_idx + 2) = src_row[src_idx];
                    *dst_row.add(dst_idx + 3) = src_row[src_idx + 3];
                }
            }
        }

        let unlock_status = unsafe {
            CVPixelBufferUnlockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::empty())
        };
        if unlock_status != kCVReturnSuccess {
            return Err(AppError::Invalid(format!(
                "CVPixelBufferUnlockBaseAddress failed: {}",
                unlock_status
            )));
        }
        Ok(())
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use std::env;
    use std::path::PathBuf;
    use std::ptr;

    use image::{codecs::jpeg::JpegEncoder, ExtendedColorType, RgbaImage};
    use libloading::Library;

    use crate::ffi::nvjpeg_bindings::{
        cudaStream_t, nvjpegEncoderParams_t, nvjpegEncoderState_t, nvjpegHandle_t, nvjpegImage_t,
        nvjpegInputFormat_t, nvjpegStatus_t, NVJPEG_INPUT_RGBI, NVJPEG_STATUS_SUCCESS,
    };
    use crate::AppError;

    const NVJPEG_DLL_CANDIDATES: [&str; 2] = ["nvjpeg64_12.dll", "nvjpeg64_11.dll"];

    pub(super) fn convert_png_to_jpeg_gpu(
        rgba: &RgbaImage,
        quality: u8,
    ) -> Result<Vec<u8>, AppError> {
        match convert_png_to_jpeg_nvjpeg(rgba, quality) {
            Ok(out) => Ok(out),
            Err(nvjpeg_err) => {
                // WHY: Windows は GPU 利用に失敗した場合に CPU へフォールバックする
                let cpu_result = convert_png_to_jpeg_cpu(rgba, quality);
                match cpu_result {
                    Ok(out) => Ok(out),
                    Err(cpu_err) => Err(AppError::Invalid(format!(
                        "nvjpeg failed: {}; cpu fallback failed: {}",
                        nvjpeg_err, cpu_err
                    ))),
                }
            }
        }
    }

    fn convert_png_to_jpeg_cpu(rgba: &RgbaImage, quality: u8) -> Result<Vec<u8>, AppError> {
        let width = rgba.width();
        let height = rgba.height();
        if width == 0 || height == 0 {
            return Err(AppError::Invalid("Invalid image size".to_string()));
        }

        let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
        for chunk in rgba.as_raw().chunks_exact(4) {
            rgb.push(chunk[0]);
            rgb.push(chunk[1]);
            rgb.push(chunk[2]);
        }

        let mut out = Vec::new();
        let quality = quality.clamp(1, 100);
        let mut encoder = JpegEncoder::new_with_quality(&mut out, quality);
        encoder.encode(&rgb, width, height, ExtendedColorType::Rgb8)?;
        Ok(out)
    }

    fn convert_png_to_jpeg_nvjpeg(
        rgba: &RgbaImage,
        quality: u8,
    ) -> Result<Vec<u8>, AppError> {
        let nvjpeg = NvJpeg::load()?;
        let mut ctx = NvJpegContext::new(&nvjpeg.fns);
        let stream: cudaStream_t = ptr::null_mut();

        unsafe {
            check_status(
                (nvjpeg.fns.create_simple)(&mut ctx.handle),
                "create handle",
            )?;
            check_status(
                (nvjpeg.fns.encoder_state_create)(ctx.handle, &mut ctx.state, stream),
                "create encoder state",
            )?;
            check_status(
                (nvjpeg.fns.encoder_params_create)(ctx.handle, &mut ctx.params, stream),
                "create encoder params",
            )?;
            let quality = quality.clamp(1, 100) as i32;
            check_status(
                (nvjpeg.fns.encoder_params_set_quality)(ctx.params, quality, stream),
                "set quality",
            )?;
        }

        let width = rgba.width() as i32;
        let height = rgba.height() as i32;
        if width <= 0 || height <= 0 {
            return Err(AppError::Invalid("Invalid image size".to_string()));
        }

        let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
        for chunk in rgba.as_raw().chunks_exact(4) {
            rgb.push(chunk[0]);
            rgb.push(chunk[1]);
            rgb.push(chunk[2]);
        }

        let mut image = nvjpegImage_t {
            channel: [ptr::null_mut(); 4],
            pitch: [0; 4],
        };
        image.channel[0] = rgb.as_mut_ptr();
        image.pitch[0] = (width as usize * 3) as u32;

        unsafe {
            check_status(
                (nvjpeg.fns.encode_image)(
                    ctx.handle,
                    ctx.state,
                    ctx.params,
                    &image,
                    NVJPEG_INPUT_RGBI,
                    width,
                    height,
                    stream,
                ),
                "encode image",
            )?;
        }

        let mut length: usize = 0;
        unsafe {
            check_status(
                (nvjpeg.fns.encode_retrieve)(ctx.handle, ctx.state, ptr::null_mut(), &mut length, stream),
                "retrieve length",
            )?;
        }
        if length == 0 {
            return Err(AppError::Invalid("nvjpeg produced empty output".to_string()));
        }

        let mut out = vec![0u8; length];
        unsafe {
            check_status(
                (nvjpeg.fns.encode_retrieve)(ctx.handle, ctx.state, out.as_mut_ptr(), &mut length, stream),
                "retrieve bitstream",
            )?;
        }
        out.truncate(length);
        Ok(out)
    }

    type NvjpegCreateSimple = unsafe extern "C" fn(*mut nvjpegHandle_t) -> nvjpegStatus_t;
    type NvjpegDestroy = unsafe extern "C" fn(nvjpegHandle_t) -> nvjpegStatus_t;
    type NvjpegEncoderStateCreate =
        unsafe extern "C" fn(nvjpegHandle_t, *mut nvjpegEncoderState_t, cudaStream_t)
            -> nvjpegStatus_t;
    type NvjpegEncoderStateDestroy = unsafe extern "C" fn(nvjpegEncoderState_t) -> nvjpegStatus_t;
    type NvjpegEncoderParamsCreate =
        unsafe extern "C" fn(nvjpegHandle_t, *mut nvjpegEncoderParams_t, cudaStream_t)
            -> nvjpegStatus_t;
    type NvjpegEncoderParamsDestroy =
        unsafe extern "C" fn(nvjpegEncoderParams_t) -> nvjpegStatus_t;
    type NvjpegEncoderParamsSetQuality =
        unsafe extern "C" fn(nvjpegEncoderParams_t, i32, cudaStream_t) -> nvjpegStatus_t;
    type NvjpegEncodeImage = unsafe extern "C" fn(
        nvjpegHandle_t,
        nvjpegEncoderState_t,
        nvjpegEncoderParams_t,
        *const nvjpegImage_t,
        nvjpegInputFormat_t,
        i32,
        i32,
        cudaStream_t,
    ) -> nvjpegStatus_t;
    type NvjpegEncodeRetrieveBitstream = unsafe extern "C" fn(
        nvjpegHandle_t,
        nvjpegEncoderState_t,
        *mut u8,
        *mut usize,
        cudaStream_t,
    ) -> nvjpegStatus_t;

    struct NvJpegFns {
        create_simple: NvjpegCreateSimple,
        destroy: NvjpegDestroy,
        encoder_state_create: NvjpegEncoderStateCreate,
        encoder_state_destroy: NvjpegEncoderStateDestroy,
        encoder_params_create: NvjpegEncoderParamsCreate,
        encoder_params_destroy: NvjpegEncoderParamsDestroy,
        encoder_params_set_quality: NvjpegEncoderParamsSetQuality,
        encode_image: NvjpegEncodeImage,
        encode_retrieve: NvjpegEncodeRetrieveBitstream,
    }

    impl NvJpegFns {
        unsafe fn load(lib: &Library) -> Result<Self, AppError> {
            unsafe {
                Ok(Self {
                    create_simple: *lib
                        .get(b"nvjpegCreateSimple\0")
                        .map_err(|e| AppError::Invalid(format!("nvjpegCreateSimple: {}", e)))?,
                    destroy: *lib
                        .get(b"nvjpegDestroy\0")
                        .map_err(|e| AppError::Invalid(format!("nvjpegDestroy: {}", e)))?,
                    encoder_state_create: *lib
                        .get(b"nvjpegEncoderStateCreate\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderStateCreate: {}", e))
                        })?,
                    encoder_state_destroy: *lib
                        .get(b"nvjpegEncoderStateDestroy\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderStateDestroy: {}", e))
                        })?,
                    encoder_params_create: *lib
                        .get(b"nvjpegEncoderParamsCreate\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderParamsCreate: {}", e))
                        })?,
                    encoder_params_destroy: *lib
                        .get(b"nvjpegEncoderParamsDestroy\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderParamsDestroy: {}", e))
                        })?,
                    encoder_params_set_quality: *lib
                        .get(b"nvjpegEncoderParamsSetQuality\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncoderParamsSetQuality: {}", e))
                        })?,
                    encode_image: *lib
                        .get(b"nvjpegEncodeImage\0")
                        .map_err(|e| AppError::Invalid(format!("nvjpegEncodeImage: {}", e)))?,
                    encode_retrieve: *lib
                        .get(b"nvjpegEncodeRetrieveBitstream\0")
                        .map_err(|e| {
                            AppError::Invalid(format!("nvjpegEncodeRetrieveBitstream: {}", e))
                        })?,
                })
            }
        }
    }

    struct NvJpeg {
        _lib: Library,
        fns: NvJpegFns,
    }

    impl NvJpeg {
        fn load() -> Result<Self, AppError> {
            let candidates = nvjpeg_library_candidates();
            let mut last_err: Option<AppError> = None;
            for candidate in candidates {
                let lib = match unsafe { Library::new(&candidate) } {
                    Ok(lib) => lib,
                    Err(err) => {
                        last_err = Some(AppError::Invalid(format!(
                            "nvjpeg library load: {}: {}",
                            candidate.display(),
                            err
                        )));
                        continue;
                    }
                };
                let fns = match unsafe { NvJpegFns::load(&lib) } {
                    Ok(fns) => return Ok(Self { _lib: lib, fns }),
                    Err(err) => {
                        last_err = Some(err);
                        continue;
                    }
                };
            }
            Err(last_err.unwrap_or_else(|| {
                AppError::Invalid("nvjpeg library load: no candidates".to_string())
            }))
        }
    }

    struct NvJpegContext<'a> {
        fns: &'a NvJpegFns,
        handle: nvjpegHandle_t,
        state: nvjpegEncoderState_t,
        params: nvjpegEncoderParams_t,
    }

    impl<'a> NvJpegContext<'a> {
        fn new(fns: &'a NvJpegFns) -> Self {
            Self {
                fns,
                handle: ptr::null_mut(),
                state: ptr::null_mut(),
                params: ptr::null_mut(),
            }
        }
    }

    impl Drop for NvJpegContext<'_> {
        fn drop(&mut self) {
            unsafe {
                if !self.params.is_null() {
                    let _ = (self.fns.encoder_params_destroy)(self.params);
                }
                if !self.state.is_null() {
                    let _ = (self.fns.encoder_state_destroy)(self.state);
                }
                if !self.handle.is_null() {
                    let _ = (self.fns.destroy)(self.handle);
                }
            }
        }
    }

    fn nvjpeg_library_candidates() -> Vec<PathBuf> {
        if let Ok(value) = env::var("NVJPEG_LIB") {
            let path = PathBuf::from(value);
            if path.is_dir() {
                return NVJPEG_DLL_CANDIDATES
                    .iter()
                    .map(|name| path.join(name))
                    .collect();
            }
            return vec![path];
        }
        NVJPEG_DLL_CANDIDATES
            .iter()
            .map(PathBuf::from)
            .collect()
    }

    fn check_status(status: nvjpegStatus_t, context: &str) -> Result<(), AppError> {
        if status == NVJPEG_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(AppError::Invalid(format!(
                "nvjpeg {} failed: {}",
                context, status
            )))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::{Mutex, OnceLock};
        use tempfile::tempdir;

        static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

        fn env_guard() -> std::sync::MutexGuard<'static, ()> {
            ENV_MUTEX.get_or_init(|| Mutex::new(())).lock().unwrap()
        }

        fn restore_env(previous: Option<String>) {
            if let Some(value) = previous {
                env::set_var("NVJPEG_LIB", value);
            } else {
                env::remove_var("NVJPEG_LIB");
            }
        }

        #[test]
        fn nvjpeg_library_candidates_uses_env_file() {
            let _guard = env_guard();
            let previous = env::var("NVJPEG_LIB").ok();
            env::set_var("NVJPEG_LIB", r"C:\fake\nvjpeg64_12.dll");
            let candidates = nvjpeg_library_candidates();
            assert_eq!(
                candidates,
                vec![PathBuf::from(r"C:\fake\nvjpeg64_12.dll")]
            );
            restore_env(previous);
        }

        #[test]
        fn nvjpeg_library_candidates_uses_env_dir() {
            let _guard = env_guard();
            let previous = env::var("NVJPEG_LIB").ok();
            let dir = tempdir().unwrap();
            env::set_var("NVJPEG_LIB", dir.path());
            let candidates = nvjpeg_library_candidates();
            let expected: Vec<PathBuf> = NVJPEG_DLL_CANDIDATES
                .iter()
                .map(|name| dir.path().join(name))
                .collect();
            assert_eq!(candidates, expected);
            restore_env(previous);
        }
    }
}
