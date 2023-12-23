// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod generate_image;

use base64;
use burn::record::{self, BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::{module::Module, tensor::backend::Backend};
use burn_wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};
use image::{ImageBuffer, Rgb, RgbImage};
use stablediffusion_wgpu::{
    model::stablediffusion::*, model_download::download_model, tokenizer::SimpleTokenizer,
};
use std::io::Cursor;
use std::process;
use tauri::State;
use tokio;

struct SDState {
    sd: StableDiffusion<Wgpu<AutoGraphicsApi, f32, i32>>,
    tokenizer: stablediffusion_wgpu::tokenizer::SimpleTokenizer,
}

fn convert_rgb_to_png(data: Vec<u8>, width: u32, height: u32) -> Vec<u8> {
    let img: RgbImage =
        ImageBuffer::from_raw(width, height, data).expect("Failed to create image buffer");

    let mut png_data = Vec::new();
    let mut cursor = Cursor::new(&mut png_data);
    img.write_to(&mut cursor, image::ImageOutputFormat::Png)
        .expect("Failed to write PNG buffer");

    cursor.into_inner().to_vec()
}

fn load_stable_diffusion_model_file<B: Backend>(
    filename: &str,
) -> Result<StableDiffusion<B>, record::RecorderError> {
    BinFileRecorder::<FullPrecisionSettings>::new()
        .load(filename.into())
        .map(|record| StableDiffusionConfig::new().init().load_record(record))
}

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn generate(prompt: &str, state: State<SDState>) -> Result<String, ()> {
    let image = generate_image::generate_image(&state.sd, &state.tokenizer, 2.5, 20, prompt);
    let image = image[0].clone();
    let png_data = convert_rgb_to_png(image, 512, 512);
    Ok(base64::encode(png_data))
}

#[tokio::main]
async fn main() {
    type Backend = Wgpu<AutoGraphicsApi, f32, i32>;
    let device = WgpuDevice::BestAvailable;

    println!("Downloading model...");
    let model_name = download_model().await.unwrap_or_else(|err| {
        println!("Error downloading model: {}", err);
        process::exit(1);
    });

    println!("Loading tokenizer...");
    let tokenizer = SimpleTokenizer::new().unwrap();
    println!("Loading model...");
    let sd: StableDiffusion<Backend> = load_stable_diffusion_model_file(model_name.as_str())
        .unwrap_or_else(|err| {
            println!("Error loading model: {}", err);
            process::exit(1);
        });
    let sd = sd.to_device(&device);

    let state = SDState {
        sd: sd,
        tokenizer: tokenizer,
    };

    println!("Starting Tauri app...");
    tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![generate])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
