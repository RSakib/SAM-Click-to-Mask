"""
SAM2 Click-to-Mask Gradio Frontend (with Mask Accumulation & Island/Hole Filtering)
---------------------------------------------------------------------------------
Interactive web UI to segment objects in images using Segment Anything 2 (SAM2).
- Drag-and-drop or click to upload your image directly into the main interactive workspace window.
- Click on the loaded image to add positive (foreground) or negative (background) points.
- Manage points, accumulate masks, clean layouts, and view a live transparent crop preview of the merged mask!
- Clicked points are displayed in real-time as green (foreground) and red (background) dots.
- Includes a dedicated Undo button to step back through click mistakes instantly by restoring cached state snapshots.
- Adjust the Logit Threshold slider (default -4.0) to make subsequent mask boundaries broader or tighter.
- Advanced controls (hidden in accordion) let you pick mask resolution scales, fill holes, or strip away stray islands.
- Merged crops are saved as standard PNGs supporting download.
"""

import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, label
from transformers import Sam2Processor, Sam2Model
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "facebook/sam2-hiera-small"

# ── Load Model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Sam2Processor.from_pretrained(MODEL_ID)
model = Sam2Model.from_pretrained(MODEL_ID).to(device)
model.eval()
print(f"Model loaded on {device}\n")


# ── Core SAM2 Helper Function ──────────────────────────────────────────────────
def run_sam2_inference(image_pil, fg_points, bg_points, mask_mode="Auto (Best predicted)", logit_threshold=-4.0):
    """
    Runs inference using the pre-loaded SAM2 model.
    Guarded to return None immediately if no points are supplied.
    """
    if not fg_points and not bg_points:
        return None

    all_points = fg_points + bg_points
    all_labels = [1] * len(fg_points) + [0] * len(bg_points)

    # Input format: batch -> object -> points -> xy
    input_points = [[all_points]]
    input_labels = [[all_labels]]

    inputs = processor(
        images=image_pil,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    W, H = image_pil.size
    num_candidates = outputs.pred_masks.shape[1]

    # Map mask modes to candidate indices
    mode_map = {
        "Sub-part (Index 0)": 0,
        "Part (Index 1)": 1,
        "Whole Object (Index 2)": 2
    }

    if mask_mode == "Auto (Best predicted)" or num_candidates == 1:
        if num_candidates == 1:
            best_idx = 0
        else:
            scores = outputs.iou_scores
            if scores.dim() == 1:
                best_idx = 0
            else:
                best_idx = int(scores[0].argmax().item())
                best_idx = min(best_idx, num_candidates - 1)
    else:
        chosen_idx = mode_map.get(mask_mode, 0)
        # Safely bound the selection within available predictions
        best_idx = min(chosen_idx, num_candidates - 1)

    raw_mask = outputs.pred_masks[0, best_idx, 0]

    # Resize raw mask logits back to original image dimensions
    mask_resized = F.interpolate(
        raw_mask.unsqueeze(0).unsqueeze(0).float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False
    ).squeeze()

    # Threshold logits (lower threshold = broader boundaries, higher = tighter boundaries)
    current_mask = (mask_resized > logit_threshold).cpu().numpy()
    return current_mask


# ── Mask Overlay Generation with Point Visualization ───────────────────────────
def generate_overlay(image_np, fg_points, bg_points, active_mask, accum_mask):
    """
    Renders the raw image with active/accumulated masks and point markers overlaid.
    """
    H, W = image_np.shape[:2]
    
    # Start with a float32 copy of the image normalized to [0, 1]
    visual_img = image_np.astype(np.float32) / 255.0

    # 1. Overlay accumulated masks in cyan/teal
    if accum_mask is not None:
        teal_color = np.array([0.1, 0.5, 0.9])
        alpha = 0.4
        visual_img[accum_mask] = (1 - alpha) * visual_img[accum_mask] + alpha * teal_color

    # 2. Overlay active current mask in green
    if active_mask is not None:
        green_color = np.array([0.1, 0.9, 0.3])
        alpha = 0.45
        visual_img[active_mask] = (1 - alpha) * visual_img[active_mask] + alpha * green_color

    # Convert back to uint8 PIL for point drawing
    overlay_pil = Image.fromarray((visual_img * 255.0).astype(np.uint8))
    draw = ImageDraw.Draw(overlay_pil)

    # Set up an adaptive radius based on image size so dots are visible but not huge
    r = max(4, int(min(W, H) * 0.008))

    # 3. Draw Foreground (Include) points in solid Green with a fine white border
    for pt in fg_points:
        x, y = pt[0], pt[1]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 230, 0), outline=(255, 255, 255), width=1)

    # 4. Draw Background (Exclude) points in solid Red with a fine white border
    for pt in bg_points:
        x, y = pt[0], pt[1]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(230, 0, 0), outline=(255, 255, 255), width=1)
    
    return overlay_pil


# ── Live Cutout Preview Generator (Saves to Temporary PNG Path) ────────────────
def get_cutout_preview(image_np, accum_mask):
    """
    Generates a tight-cropped transparent preview (RGBA) of only the accumulated mask.
    Saves it to a temporary PNG file path to guarantee the download formats as a PNG image.
    """
    if image_np is None or accum_mask is None:
        return None

    H, W = image_np.shape[:2]

    # Build transparent canvas
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = image_np
    rgba[accum_mask, 3] = 255

    # Crop tight to bounding box
    rows = np.any(accum_mask, axis=1)
    cols = np.any(accum_mask, axis=0)

    if not rows.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = rgba[rmin : rmax + 1, cmin : cmax + 1]

    # Render crop
    cropped_pil = Image.fromarray(cropped, mode="RGBA")
    
    # Save the file to a temporary location with a .png extension so downloading respects the format
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "sam2_cutout_crop.png")
    cropped_pil.save(temp_file_path, "PNG")

    # Return the file path so Gradio serves it with correct metadata & explicit download structure
    return temp_file_path


# ── Connected-Component Clean Helper (Remove Stray Islands) ───────────────────
def remove_stray_islands(mask, min_size=200):
    """
    Labels separate connected objects/islands in the boolean mask and
    filters out any components that are smaller than the specified pixel limit.
    """
    if mask is None:
        return None
    
    labeled_array, num_features = label(mask)
    if num_features <= 1:
        # 0 features is empty; 1 feature means there are no extra islands to remove
        return mask

    # Count pixels associated with each unique label
    label_sizes = np.bincount(labeled_array.ravel())
    
    # Filter labels: keep only components exceeding the pixel threshold (and ignore background index 0)
    keep_labels = label_sizes > min_size
    keep_labels[0] = False
    
    cleaned_mask = keep_labels[labeled_array]
    return cleaned_mask


# ── State Initialization ───────────────────────────────────────────────────────
def get_empty_state():
    return {
        "fg_points": [],       # List of [x, y]
        "bg_points": [],       # List of [x, y]
        "active_mask": None,   # Boolean numpy array
        "accum_mask": None,    # Boolean numpy array
        "image_np": None,      # Raw numpy array
        "image_pil": None,     # Loaded PIL Image
        "history": []          # History stack storing prior state dictionaries (snapshots)
    }


# ── Event Processing Callbacks ────────────────────────────────────────────────
def on_image_upload(image_pil):
    """
    Fires only when a user drops or selects a file within the workspace panel.
    Initializes a clean segmenting state with the new image.
    """
    if image_pil is None:
        return get_empty_state(), None
    
    state = get_empty_state()
    state["image_pil"] = image_pil
    state["image_np"] = np.array(image_pil)
    
    return state, None


def on_image_clear():
    """
    Fires when the user clicks 'x' to clear the image window.
    """
    return get_empty_state(), None


def on_image_click(state, point_type, mask_mode, logit_threshold, evt: gr.SelectData):
    if state["image_pil"] is None:
        return None, state

    # Gradio SelectData provides [x, y] on the original coordinates
    x, y = evt.index[0], evt.index[1]

    # Save a snapshot of the *prior* active state before modifying anything
    snapshot = {
        "fg_points": list(state["fg_points"]),
        "bg_points": list(state["bg_points"]),
        "active_mask": state["active_mask"].copy() if state["active_mask"] is not None else None
    }
    state["history"].append(snapshot)

    # Apply the new point click
    if point_type == "Foreground (Add)":
        state["fg_points"].append([x, y])
    else:
        state["bg_points"].append([x, y])

    # Run SAM2 Inference with current selection coordinates and logit threshold
    active_mask = run_sam2_inference(
        state["image_pil"], 
        state["fg_points"], 
        state["bg_points"],
        mask_mode,
        logit_threshold
    )
    state["active_mask"] = active_mask

    # Re-draw the visualization overlay with live point dots overlaid on the view
    overlay = generate_overlay(
        state["image_np"],
        state["fg_points"],
        state["bg_points"],
        state["active_mask"],
        state["accum_mask"]
    )

    return overlay, state


def handle_undo_point(state):
    """
    Pops the most recent active state snapshot from history and restores it instantly.
    If there is no active mask, or if only 1 or 0 points exist, it defaults to behaving
    exactly like resetting the active mask (handle_reset_active) to avoid unnecessary recalculations.
    """
    if state["image_pil"] is None:
        return None, state

    # Calculate current total plotted active points
    num_points = len(state["fg_points"]) + len(state["bg_points"])

    # If no active mask exists, or if there is 1 or fewer active points, clear active state immediately.
    if state["active_mask"] is None or num_points <= 1:
        return handle_reset_active(state)

    if not state["history"]:
        # If no history exists, render current visual state
        overlay = generate_overlay(
            state["image_np"],
            state["fg_points"],
            state["bg_points"],
            state["active_mask"],
            state["accum_mask"]
        )
        return overlay, state

    # Pop the last saved state snapshot and restore it directly
    snapshot = state["history"].pop()
    state["fg_points"] = snapshot["fg_points"]
    state["bg_points"] = snapshot["bg_points"]
    state["active_mask"] = snapshot["active_mask"]

    # Redraw overlay immediately using the restored state details
    overlay = generate_overlay(
        state["image_np"],
        state["fg_points"],
        state["bg_points"],
        state["active_mask"],
        state["accum_mask"]
    )

    return overlay, state


def handle_accumulate(state):
    if state["active_mask"] is None:
        overlay = generate_overlay(state["image_np"], [], [], None, state["accum_mask"])
        cutout_path = get_cutout_preview(state["image_np"], state["accum_mask"])
        return overlay, state, cutout_path

    # Merge active mask into master accumulated mask
    if state["accum_mask"] is None:
        state["accum_mask"] = state["active_mask"].copy()
    else:
        state["accum_mask"] = state["accum_mask"] | state["active_mask"]

    # Clear active points, active mask, and local active history stack
    state["fg_points"].clear()
    state["bg_points"].clear()
    state["history"].clear()
    state["active_mask"] = None

    overlay = generate_overlay(
        state["image_np"],
        state["fg_points"],
        state["bg_points"],
        state["active_mask"],
        state["accum_mask"]
    )
    
    # Generate updated live preview cutout path (.png)
    cutout_path = get_cutout_preview(state["image_np"], state["accum_mask"])
    return overlay, state, cutout_path


def handle_fill_accumulated(state):
    if state["accum_mask"] is not None:
        state["accum_mask"] = binary_fill_holes(state["accum_mask"])
    
    overlay = generate_overlay(
        state["image_np"],
        state["fg_points"],
        state["bg_points"],
        state["active_mask"],
        state["accum_mask"]
    )
    
    # Since accumulated mask changed, update the cutout preview (.png)
    cutout_path = get_cutout_preview(state["image_np"], state["accum_mask"])
    return overlay, state, cutout_path


def handle_remove_accumulated_islands(state):
    if state["accum_mask"] is not None:
        # Filters out connected islands smaller than 200 pixels
        state["accum_mask"] = remove_stray_islands(state["accum_mask"], min_size=200)
        
    overlay = generate_overlay(
        state["image_np"],
        state["fg_points"],
        state["bg_points"],
        state["active_mask"],
        state["accum_mask"]
    )
    
    # Since accumulated mask changed, update the cutout preview (.png)
    cutout_path = get_cutout_preview(state["image_np"], state["accum_mask"])
    return overlay, state, cutout_path


def handle_reset_active(state):
    """
    Clears all active state variables, points, current active masks, and the undo queue.
    This safely returns the clean updated visual overlay immediately with only the accumulated 
    mask visible, without running any SAM2 model inference.
    """
    state["fg_points"].clear()
    state["bg_points"].clear()
    state["history"].clear()
    state["active_mask"] = None
    
    overlay = generate_overlay(
        state["image_np"],
        state["fg_points"],
        state["bg_points"],
        state["active_mask"],
        state["accum_mask"]
    )
    return overlay, state


def handle_clear_all(state):
    cleared = get_empty_state()
    if state["image_pil"] is not None:
        cleared["image_pil"] = state["image_pil"]
        cleared["image_np"] = state["image_np"]
        return cleared["image_pil"], cleared, None
    return None, cleared, None


# ── Gradio Theme & Layout ──────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber")) as demo:
    # App State
    state = gr.State(get_empty_state)

    gr.Markdown(
        """
        # 🟢 SAM2 Interactive Click-to-Mask Tool
        Drag & drop or upload an image directly into the workspace below. Click to define active segment masks (green = include, red = exclude), then press **Accumulate** to lock them into your final transparent crop.
        """
    )

    with gr.Row():
        # Left Side - Integrated Upload & Interaction Workspace
        with gr.Column(scale=3):
            # Combined Image Window (acting as uploader when empty and workspace canvas when loaded)
            image_display = gr.Image(
                label="Image Workspace (Upload & Click to Segment)",
                interactive=True,
                show_label=True,
                type="pil"
            )
            
            with gr.Row():
                point_type = gr.Radio(
                    choices=["Foreground (Add)", "Background (Exclude)"],
                    value="Foreground (Add)",
                    label="Point Selection Mode",
                    info="Choose whether your next click represents an inclusion or exclusion point."
                )

        # Right Side - Live Cutout Preview & Operations
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Live Cutout Preview (Merged)")
            # Transparent Cutout Viewer - Setting type to filepath guarantees clean download handling
            cutout_preview_display = gr.Image(
                label="Merged Transparent Crop (Supports direct PNG download)",
                interactive=False,
                type="filepath"
            )

            gr.Markdown("### ⚙️ Mask Adjustments & Operations")
            
            # Logit threshold slider (negative values expand, positive values contract the mask boundary)
            logit_threshold = gr.Slider(
                minimum=-10.0,
                maximum=10.0,
                value=-4.0,  # Default threshold set to -4.0
                step=0.1,
                label="Logit Threshold",
                info="Lower values expand the mask (broader); higher values shrink it (tighter). Will apply on your next click."
            )

            with gr.Group():
                accumulate_btn = gr.Button("➕ Accumulate (Merge) Current Mask", variant="primary")
                undo_btn = gr.Button("↩️ Undo Last Point", variant="secondary")
                reset_btn = gr.Button("🧹 Clear Active Mask", variant="secondary")
                clear_btn = gr.Button("🧹 Clear All Mask Progress", variant="stop")
            
            with gr.Accordion("Advanced Mask Cleaning & Controls", open=False):
                # Mask Prediction Scale Selection Dropdown (Now placed as an advanced setting)
                mask_mode = gr.Dropdown(
                    choices=[
                        "Auto (Best predicted)", 
                        "Sub-part (Index 0)", 
                        "Part (Index 1)", 
                        "Whole Object (Index 2)"
                    ],
                    value="Auto (Best predicted)",
                    label="Mask Resolution Level",
                    info="Specify whether the mask should cover a micro-component, part, or the entire broad object."
                )
                fill_accum_btn = gr.Button("🕳️ Fill Holes inside Accumulated Mask")
                remove_islands_btn = gr.Button("🏝️ Remove Stray Islands in Accumulated Mask")

    # ── Wire Up Event Streams ─────────────────────────────────────────────────
    
    # 1. Image upload - Sets session state when uploader gets a file (does not re-evaluate outputs programmatically)
    image_display.upload(
        fn=on_image_upload,
        inputs=[image_display],
        outputs=[state, cutout_preview_display]
    )

    # 2. Image clear - Resets session state when uploader clears the file
    image_display.clear(
        fn=on_image_clear,
        outputs=[state, cutout_preview_display]
    )

    # 3. Click interactions (updates the main display overlay, respects selected parameters)
    image_display.select(
        fn=on_image_click,
        inputs=[state, point_type, mask_mode, logit_threshold],
        outputs=[image_display, state]
    )

    # 4. Undo action (pops last added point snapshot state, restoring visual mask instantly without model inference)
    undo_btn.click(
        fn=handle_undo_point,
        inputs=[state],
        outputs=[image_display, state]
    )

    # 5. Mask accumulation (merges active selection and updates the live cutout preview)
    accumulate_btn.click(
        fn=handle_accumulate,
        inputs=[state],
        outputs=[image_display, state, cutout_preview_display]
    )

    # 6. Clean active selection (does not touch the live preview showing accumulated segments)
    reset_btn.click(
        fn=handle_reset_active,
        inputs=[state],
        outputs=[image_display, state]
    )

    # 7. Full Clear Reset (clears overlay and live preview)
    clear_btn.click(
        fn=handle_clear_all,
        inputs=[state],
        outputs=[image_display, state, cutout_preview_display]
    )

    # 8. Fill closed accumulated holes (updates the live preview as it changes the accumulated mask)
    fill_accum_btn.click(
        fn=handle_fill_accumulated,
        inputs=[state],
        outputs=[image_display, state, cutout_preview_display]
    )

    # 9. Remove stray islands in the accumulated mask (updates the live preview with cleaned mask)
    remove_islands_btn.click(
        fn=handle_remove_accumulated_islands,
        inputs=[state],
        outputs=[image_display, state, cutout_preview_display]
    )


if __name__ == "__main__":
    demo.launch()