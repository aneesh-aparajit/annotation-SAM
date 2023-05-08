import json
import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

MODEL_WEIGHT_CKPT = "../weights/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cpu"

sam_model = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_WEIGHT_CKPT)
model = SamPredictor(sam_model=sam_model)

annotations = []
selected_indices = []
rgb_mask = None
gray_mask = None


def hex_to_rgb(h):
    h = h[1:]
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def save_progress(pixel_id, class_id, class_label, color_map, mask, threshold) -> None:
    global annotations, rgb_mask, gray_mask
    color_map = hex_to_rgb(color_map)
    rgb = np.array(mask)
    gray = np.array(mask)[:, :, 0]
    mask = np.array(mask)[:, :, 0]
    mask[mask > threshold] = 255
    if gray_mask is None:
        gray_mask = np.zeros_like(gray)
        gray_mask[mask == 255] = pixel_id
    else:
        gray_mask[mask == 255] = pixel_id

    if rgb_mask is None:
        rgb_mask = np.zeros_like(rgb)
        rgb_mask[mask == 255, :] = color_map
    else:
        rgb_mask[mask == 255, :] = color_map

    # print(np.unique(full_mask))
    annotations.append(
        {
            "class_id": class_id,
            "class_label": class_label,
            "pixel_id": pixel_id,
            "rgb_color": color_map,
        }
    )
    selected_indices.clear()
    __import__("pprint").pprint(annotations)
    return Image.fromarray(gray_mask), Image.fromarray(rgb_mask)


def on_click_input_image(image, evt: gr.SelectData) -> Image:
    global selected_indices
    image = np.array(image)
    selected_indices.append(evt.index)
    print(selected_indices)
    model.set_image(image=image)

    input_points = np.array(selected_indices)
    input_labels = np.ones(shape=(input_points.shape[0]))
    masks, _, _ = model.predict(
        point_coords=input_points, point_labels=input_labels, multimask_output=False
    )
    output_image = Image.fromarray(masks[0, :, :])
    return output_image


def save_meta(annotation_dir: str) -> None:
    if not os.path.exists('../annotations/'):
        os.makedirs('../annotations/')
    if not os.path.exists(f"../annotations/{annotation_dir}"):
        os.makedirs(f"../annotations/{annotation_dir}", exist_ok=True)
    with open(f"../annotations/{annotation_dir}/annot.json", "w") as fout:
        json.dump(annotations, fout)
    plt.imsave(f"../annotations/{annotation_dir}/rgb_mask.png", rgb_mask)
    plt.imsave(
        f"../annotations/{annotation_dir}/gray_mask.png", gray_mask, cmap="gray"
    )
    return None


with gr.Blocks() as demo:
    gr.Markdown(
        """# Image Annotation Tool Powered by SAM
## How to use?
- Click on a portion of the image which you want to segment.
- Add the required details in the Class ID and Class Label boxes.
    - Please ensure there are uniformity in what you input
- To save the current label, click on the save progress to save the label.
"""
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
        with gr.Column():
            output_image = gr.Image(label="Masked Image", type="pil")
    with gr.Row():
        with gr.Column():
            gray_image = gr.Image(label="Grayscale Masks", type="pil")
        with gr.Column():
            color_image = gr.Image(label="RGB Masks", type="pil")
    with gr.Row():
        with gr.Column():
            pixel_id = gr.Number(value=1, label="Pixel ID")
        with gr.Column():
            class_name = gr.Textbox(label="Class Label")
        with gr.Column():
            class_id = gr.Number(label=" Class ID")
    with gr.Row():
        with gr.Column():
            color = gr.ColorPicker(label="Choose the color for the segmentation map")
        with gr.Column():
            annot_dir_name = gr.Textbox(label="Enter Annotation Folder Name")
        with gr.Column():
            threshold = gr.Number(value=200, label="Mask Threshold")

    input_image.select(
        fn=on_click_input_image,
        inputs=[input_image],
        outputs=[output_image],
    )
    with gr.Row():
        btn = gr.Button(value="Save Progress!")
        btn.click(
            fn=save_progress,
            inputs=[pixel_id, class_id, class_name, color, output_image, threshold],
            outputs=[gray_image, color_image],
        )

        save = gr.Button(value="Save Image/Color Maps")
        save.click(fn=save_meta, inputs=[annot_dir_name], outputs=None)


if __name__ == "__main__":
    demo.launch()
