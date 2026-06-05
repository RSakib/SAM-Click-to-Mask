# 🟢 Click-to-Mask: SAM2 Interactive Segmentation Tool

**Click-to-Mask** is essentially a better Magic Wand tool that is powered by Meta's **Segment Anything 2 (SAM2)** model. This interactive Gradio web application lets you instantly isolate, accumulate, clean, and export transparent PNG object layers simply by clicking on the image that you upload. This is best used when trying to separate a focal subject from a busy background.

### 🌐 [Try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/RSakib/SAM-interactive-click-to-PNG)

---

## ✨ Key Features

* **AI-Powered Precision:** Leverages `facebook/sam2-hiera-small` to intelligently snap to object boundaries, vastly outperforming legacy pixel-color magic wands.
* **Interactive Click-to-Mask:** 
    * 🟢 **Foreground (Add):** Drop green points to tell the model what to *include*.
    * 🔴 **Background (Exclude):** Drop red points to tell the model what to *exclude*.
* **Mask Accumulation:** Piece together complex selections! Mask one part of an image, click **Accumulate**, and seamlessly start masking the next piece without losing your progress.
* **Instant Undo History:** Made a bad click? The dedicated **Undo** button lets you step backward instantly through your click history by restoring cached state snapshots.
* **Advanced Mask Cleanup:** 
    * 🕳️ **Fill Holes:** Instantly patch up hollow spaces or accidental gaps inside your accumulated mask.
    * 🏝️ **Remove Stray Islands:** Strip away tiny, unintended background artifacts with a single click.
* **Live Cutout Preview:** View a real-time, tightly cropped transparent preview of your isolated object, saved as a standard PNG ready for direct download.
