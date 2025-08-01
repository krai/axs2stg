import logging
import os
from typing import Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import backend
from elastic_models.diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler
import types

os.environ["THESTAGE_AUTH_TOKEN"] = "eyJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJkZWNiZjQyZi1kZTAyLTRjZWEtYmU1MC1lMDU0NWY0OWJlNjgiLCJzdWIiOiJhcGlVc2VyIiwidXNlcklkIjoxNzgsImNsaWVudElkIjoxODksInRva2VuVHlwZSI6ImFjY2VzcyIsIm9hdXRoQ2xpZW50SWQiOiJUSEVTVEFHRV9DTEkiLCJleHAiOjE3ODUyNDQyNzB9.4zFJW_GVZBiJCUkbCngVAOyUsJU5WSEuISZHhFByiZo"

log = None # logging.getLogger("backend-stageai")

torch.set_float32_matmul_precision('medium')


def upcast_vae(self):
    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

    if needs_upcasting:
        self.upcast_vae()
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != self.vae.dtype:
        if torch.backends.mps.is_available():
            self.vae = self.vae.to(latents.dtype)

def decode_latent(self, latents):

    has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
    has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        )
        latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / self.vae.config.scaling_factor

    image = self.vae.decode(latents, return_dict=False)[0]

    return image

from itertools import chain

def new_call(self, *args, **kwargs):
    if 'prompt' in kwargs:
        bs = len(kwargs['prompt'])
    else:
        bs = kwargs['prompt_embeds'].shape[0]

    BS_decode = 4

    if bs % BS_decode != 0 or kwargs['output_type'] == 'latent':
        return self(*args, **kwargs).images


    kwargs['output_type'] = 'latent'
    latent = self(*args, **kwargs).images
    output = []

    for l in range(0, bs, BS_decode):
        image = self.decode_latent(latent[l:l+BS_decode])
        image = self.image_processor.postprocess(image, output_type='pt')
        output.append(image)

    return chain(*output)


class BackendStageai(backend.Backend):
    def __init__(
            self,
            model_name='stabilityai/stable-diffusion-xl-base-1.0',
            hf_token=None,
            device='cuda',
            precision='fp16',
            mode='XL',
            guidance=8,
            steps=20,
            negative_prompt="normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
            latent_dir='/opt/dlami/nvme/axs2stg-dev/text_to_image/tools/latents.pt',
            batch_size=1,
            cache_dir=None
        ):
        super().__init__()
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device if torch.cuda.is_available() else 'cpu'
        if precision == 'fp16':
            self.dtype = torch.float16
        elif precision == 'bf16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.mode = mode
        self.guidance = guidance
        self.steps = steps
        self.negative_prompt = negative_prompt
        self.latent_dir = latent_dir
        self.batch_size = batch_size
        self.max_length_neg_prompt = 77
        self.cache_dir = cache_dir
        global log
        log = logging.getLogger(f"backend-stageai-{self.device}")

    def version(self):
        return torch.__version__

    def name(self):
        return "stageai-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        log.info(f"Loading StableDiffusionXL Pipeline {self.model_name} on {self.device}")

        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_name, subfolder="scheduler"
        )


        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            mode="S",
            cache_dir=self.cache_dir,
            scheduler=self.scheduler,
            token=self.hf_token,
            device_map=self.device,
        )

        self.pipe.set_progress_bar_config(disable=True)
        
        # Explicitly move to the correct device
        self.pipe.to(self.device)

        self.pipe.decode_latent = types.MethodType(decode_latent, self.pipe)
        self.pipe.upcast_vae()
        self.pipe.vae.decoder.to(memory_format=torch.channels_last)
        self.pipe.new_call = types.MethodType(new_call, self.pipe)

        self.pipe.decode_latent = torch.compile(self.pipe.decode_latent, mode="max-autotune")
        self.encode_tokens_compiled = torch.compile(self.encode_tokens_impl, mode="max-autotune")

        # Load latents from file
        log.info(f"Loading latents from {self.latent_dir}")
        self.latents = torch.load(self.latent_dir, map_location=self.device)
        self.latents = self.latents.to(dtype=torch.float16, device=self.device)
        log.info(f"Loaded latents shape: {self.latents.shape}")

        # Pre-tokenize negative prompt
        self.negative_prompt_tokens = self.pipe.tokenizer(
            self.convert_prompt(self.negative_prompt, self.pipe.tokenizer),
            padding="max_length",
            max_length=self.max_length_neg_prompt,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        self.negative_prompt_tokens_2 = self.pipe.tokenizer_2(
            self.convert_prompt(self.negative_prompt, self.pipe.tokenizer_2),
            padding="max_length",
            max_length=self.max_length_neg_prompt,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        return self

    def tokenize_captions(self, captions):
        """Tokenize captions and return input dictionaries."""
        inputs = []
        for caption in captions:
            # Tokenize with both tokenizers
            tokens_1 = self.pipe.tokenizer(
                self.convert_prompt(caption, self.pipe.tokenizer),
                padding="max_length",
                max_length=self.max_length_neg_prompt,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            tokens_2 = self.pipe.tokenizer_2(
                self.convert_prompt(caption, self.pipe.tokenizer_2),
                padding="max_length",
                max_length=self.max_length_neg_prompt,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            
            inputs.append({
                "input_tokens": tokens_1,
                "input_tokens_2": tokens_2,
            })
        
        return inputs

    def convert_prompt(self, prompt, tokenizer):
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1
                prompt = prompt.replace(token, replacement)
        return prompt

    def encode_tokens(
        self,
        pipe,
        text_input: torch.Tensor,
        text_input_2: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[torch.Tensor] = None,
        negative_prompt_2: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
    ):
        if text_input.shape[0] == self.batch_size:
            return self.encode_tokens_compiled(
                pipe,
                text_input,
                text_input_2,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                clip_skip
            )
        else:
            return self.encode_tokens_impl(
                pipe,
                text_input,
                text_input_2,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                clip_skip
            )

    def encode_tokens_impl(
        self,
        pipe,
        text_input: torch.Tensor,
        text_input_2: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[torch.Tensor] = None,
        negative_prompt_2: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
    ):
        """Encodes the input tokens into text encoder hidden states."""
        device = self.device #or pipe._execution_device
        batch_size = text_input.shape[0]

        # Define tokenizers and text encoders
        tokenizers = (
            [pipe.tokenizer, pipe.tokenizer_2]
            if pipe.tokenizer is not None
            else [pipe.tokenizer_2]
        )
        text_encoders = (
            [pipe.text_encoder, pipe.text_encoder_2]
            if pipe.text_encoder is not None
            else [pipe.text_encoder_2]
        )

        if prompt_embeds is None:
            text_input_2 = text_input_2 if text_input_2 is not None else text_input

            prompt_embeds_list = []
            text_inputs_list = [text_input, text_input_2]
            for text_inputs, tokenizer, text_encoder in zip(
                text_inputs_list, tokenizers, text_encoders
            ):
                text_input_ids = text_inputs
                prompt_embeds = text_encoder(
                    text_input_ids.to(device), output_hidden_states=True
                )

                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # Get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = (
            negative_prompt is None and pipe.config.force_zeros_for_empty_prompt
        )
        if (
            do_classifier_free_guidance
            and negative_prompt_embeds is None
            and zero_out_negative_prompt
        ):
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            #negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 if negative_prompt_2 is not None else negative_prompt

            negative_prompt_inputs = (
                negative_prompt.repeat(batch_size, 1)
                #if (len(negative_prompt.shape) == 1)
                #else negative_prompt
            )
            negative_prompt_2_inputs = (
                negative_prompt_2.repeat(batch_size, 1)
                #if (len(negative_prompt_2.shape) == 1)
                #else negative_prompt_2
            )

            uncond_inputs = [negative_prompt_inputs, negative_prompt_2_inputs]

            negative_prompt_embeds_list = []
            for uncond_input, tokenizer, text_encoder in zip(
                uncond_inputs, tokenizers, text_encoders
            ):
                negative_prompt_embeds = text_encoder(
                    uncond_input.to(device),
                    output_hidden_states=True,
                )
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if pipe.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=pipe.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            if pipe.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=pipe.text_encoder_2.dtype, device=device
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=pipe.unet.dtype, device=device
                )

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                1, num_images_per_prompt
            ).view(bs_embed * num_images_per_prompt, -1)
            
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_inputs(self, inputs, i):
        """Prepare inputs for a batch, encoding tokens into embeddings."""
        if self.batch_size == 1:
            prompt = inputs[i]["input_tokens"]
            prompt_2 = inputs[i]["input_tokens_2"]
        else:
            prompt = []
            prompt_2 = []
            for inp in inputs[i: min(i + self.batch_size, len(inputs))]:
                prompt.append(inp["input_tokens"])
                prompt_2.append(inp["input_tokens_2"])

            prompt = torch.cat(prompt)
            prompt_2 = torch.cat(prompt_2)

        return self.encode_tokens(
            self.pipe,
            prompt,
            prompt_2,
            negative_prompt=self.negative_prompt_tokens,
            negative_prompt_2=self.negative_prompt_tokens_2,
        )
    
    def warmup(self):
        inputs = ["krazy people working on MLPerf Submission"] 

        self.pipe.set_progress_bar_config(disable=False, desc=f"Warming up on {self.device}")
        for bs in [self.batch_size] * 3:
            self.predict(inputs*bs)
        self.pipe.set_progress_bar_config(disable=True)

    def predict(self, inputs):
        """
        Predict method that accepts either strings (for backward compatibility)
        or dictionaries with tokenized inputs.
        """
        images = []
        
        # Check if inputs are strings (captions) or dictionaries (tokenized)
        if isinstance(inputs[0], str):
            # Tokenize captions
            inputs = self.tokenize_captions(inputs)
        
        batch_size = len(inputs)
        
        with torch.no_grad():
            for i in range(0, len(inputs), self.batch_size):
                # Prepare latents for this batch
                batch_size_actual = min(self.batch_size, len(inputs) - i)
                latent = self.latents
                if latent.dim() == 4:
                    if latent.shape[0] < batch_size_actual:
                        latent = latent.repeat(batch_size_actual, 1, 1, 1)
                    else:
                        latent = latent[:batch_size_actual]
                
                # Prepare embeddings
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.prepare_inputs(inputs, i)

                # Run pipeline with embeddings
                results = self.pipe.new_call(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    guidance_scale=self.guidance,
                    num_inference_steps=self.steps,
                    latents=latent,
                    output_type='pt',
                )
                
                for img in results:
                    images.append(img)

        return images
