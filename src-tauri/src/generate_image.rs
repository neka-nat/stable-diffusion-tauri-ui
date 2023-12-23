use stablediffusion_wgpu::{model::stablediffusion::*, tokenizer::SimpleTokenizer};

use burn_wgpu::{AutoGraphicsApi, Wgpu};

pub fn generate_image(
    sd: &StableDiffusion<Wgpu<AutoGraphicsApi, f32, i32>>,
    tokenizer: &SimpleTokenizer,
    unconditional_guidance_scale: f64,
    n_steps: usize,
    prompt: &str,
) -> Vec<Vec<u8>> {
    let unconditional_context = sd.unconditional_context(&tokenizer);
    let context = sd.context(&tokenizer, prompt).unsqueeze::<3>(); //.repeat(0, 2); // generate 2 sample//s

    println!("Sampling image...");
    let images = sd.sample_image(
        context,
        unconditional_context,
        unconditional_guidance_scale,
        n_steps,
    );
    images
}
