from contextlib import nullcontext
import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"
context = autocast if device == "cuda" else nullcontext
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained("dainlee/output_final", torch_dtype=dtype)
pipe = pipe.to(device)


# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = True

if disable_safety:
  def null_safety(images, **kwargs):
      return images, False
  pipe.safety_checker = null_safety


def infer(prompt, n_samples, steps, scale):

    with context("cuda"):
        images = pipe(n_samples*[prompt], guidance_scale=scale, num_inference_steps=steps).images

    return images

css = """
        a {
            color: inherit;
            text-decoration: underline;
        }
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: #9d66e5;
            background: #9d66e5;
        }
        input[type='range'] {
            accent-color: #9d66e5;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-options {
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .logo{ filter: invert(1); }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
"""

block = gr.Blocks(css=css)

examples = [
    [
        'Yoda',
        2,
        7.5,
    ],
    [
        'Abraham Lincoln',
        2,
        7.5,
    ],
    [
        'George Washington',
        2,
        7,
    ],
]

with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div>
                <h1 style="font-weight: 900; font-size: 3rem;">
                  One piece text to image
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
              Generate new one piece character from a text description,
                <a href=>created by YaYaB</a>.
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")


        with gr.Row(elem_id="advanced-options"):
            samples = gr.Slider(label="Images", minimum=1, maximum=4, value=2, step=1)
            steps = gr.Slider(label="Steps", minimum=5, maximum=50, value=25, step=5)
            scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
            )


        ex = gr.Examples(examples=examples, fn=infer, inputs=[text, samples, scale], outputs=gallery, cache_examples=False)
        ex.dataset.headers = [""]


        text.submit(infer, inputs=[text, samples, steps, scale], outputs=gallery)
        btn.click(infer, inputs=[text, samples, steps, scale], outputs=gallery)
        gr.HTML(
            """
                <div class="footer">
                    <p> Gradio Demo by 🤗 Hugging Face and YaYaB
                    </p>
                </div>
           """
        )

block.launch()