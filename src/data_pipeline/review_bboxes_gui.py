"""Manual bbox annotation/correction tool. Run this yourself (needs a
display -- Tkinter):

    python src/data_pipeline/review_bboxes_gui.py

Reads data/vision_dataset/manifest.jsonl (written by
collect_vision_dataset.py) -- one entry per UNIQUE id actually collected
into the per-head/per-class folders, each with its existing CuneiML bbox if
one was recorded, or none. For each id, shows the photo with the current
bbox in red if there is one, or no box at all if there isn't. Controls:
  - Drag on the image to draw a box (shown in green) -- required if there
    was no existing box, optional (as a correction) if there was.
  - "Keep / Save" -- saves whichever box is currently shown (red if you
    didn't redraw, green if you did) and moves to the next image. Does
    nothing if there is no box at all yet (draw one, or use "No tablet").
  - "No tablet visible" -- marks this id as unusable (no bbox) and moves on.
  - "Skip (decide later)" -- leaves it out of the corrections file entirely,
    moves on without recording anything.

Progress is saved incrementally to data/bbox_corrections.jsonl (one JSON
line per decision: {"id", "bbox" or null, "status"}). Closing the window and
re-running the script resumes after the last reviewed id -- already-decided
ids are skipped automatically. Since a decision is keyed by id (not by which
class folder it lives in), reviewing an id once covers every class folder
it was collected into.
"""
import json
import os
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FULL_IMG_DIR = os.path.join(BASE_DIR, "data", "raw", "cuneiml", "images_full")
VISION_DATASET_DIR = os.path.join(BASE_DIR, "data", "vision_dataset")
MANIFEST_FILE = os.path.join(VISION_DATASET_DIR, "manifest.jsonl")
CORRECTIONS_FILE = os.path.join(BASE_DIR, "data", "bbox_corrections.jsonl")
MAX_DISPLAY = 850  # canvas fits inside this many px on the longer side


def build_path_index():
    """id -> a filesystem path holding that id's image (any one copy)."""
    index = {}
    if os.path.isdir(FULL_IMG_DIR):
        for fn in os.listdir(FULL_IMG_DIR):
            index[fn.rsplit(".", 1)[0]] = os.path.join(FULL_IMG_DIR, fn)
    for root, _dirs, files in os.walk(VISION_DATASET_DIR):
        for fn in files:
            if fn.endswith(".jpg"):
                pid = fn.rsplit(".", 1)[0]
                index.setdefault(pid, os.path.join(root, fn))
    return index


def load_manifest():
    items = []
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


class ReviewApp:
    def __init__(self, root, items, done_ids, path_index):
        self.root = root
        self.path_index = path_index
        self.items = [it for it in items if str(it["id"]) not in done_ids]
        self.total_all = len(items)
        self.total_done = len(done_ids)
        self.idx = 0

        self.root.title("Bbox review")
        self.status_label = ttk.Label(root, text="", font=("Segoe UI", 11))
        self.status_label.pack(pady=4)

        self.canvas = tk.Canvas(root, width=MAX_DISPLAY, height=MAX_DISPLAY, bg="black")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="Keep / Save  (Enter)", command=self.save_current).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="No tablet visible  (N)", command=self.mark_bad).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Skip (decide later)  (S)", command=self.skip).grid(row=0, column=2, padx=5)

        root.bind("<Return>", lambda e: self.save_current())
        root.bind("n", lambda e: self.mark_bad())
        root.bind("s", lambda e: self.skip())

        self.corrections_f = open(CORRECTIONS_FILE, "a", encoding="utf-8")

        self.drag_start = None
        self.rect_id = None
        self.current_box_canvas = None  # (x1,y1,x2,y2) in canvas coords, or None
        self.scale = 1.0
        self.img_tk = None

        self.load_current()

    def current_item(self):
        return self.items[self.idx]

    def load_current(self):
        if self.idx >= len(self.items):
            self.status_label.config(text="All done!")
            self.canvas.delete("all")
            return
        item = self.current_item()
        pid = str(item["id"])
        path = self.path_index.get(pid)
        if path is None:
            # shouldn't happen, but don't crash the whole session over one bad id
            self.advance()
            return
        img = Image.open(path).convert("RGB")
        w, h = img.size
        self.scale = min(MAX_DISPLAY / w, MAX_DISPLAY / h, 1.0)
        disp_w, disp_h = int(w * self.scale), int(h * self.scale)
        disp_img = img.resize((disp_w, disp_h))
        self.img_tk = ImageTk.PhotoImage(disp_img)

        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)

        bbox = item.get("bbox")
        if bbox:
            (x1, y1), (x2, y2) = bbox
            cx1, cy1, cx2, cy2 = x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale
            self.rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="red", width=3)
            self.current_box_canvas = (cx1, cy1, cx2, cy2)
        else:
            self.rect_id = None
            self.current_box_canvas = None

        done_so_far = self.total_done + self.idx
        hint = "drag to redraw" if bbox else "no box yet -- drag to draw one"
        self.status_label.config(
            text=f"{done_so_far}/{self.total_all}  |  id={pid}  |  {hint}, red=existing, green=yours"
        )

    def on_press(self, event):
        self.drag_start = (event.x, event.y)

    def on_drag(self, event):
        if self.drag_start is None:
            return
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        x0, y0 = self.drag_start
        self.rect_id = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="lime", width=3)

    def on_release(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.current_box_canvas = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.drag_start = None

    def _write(self, pid, bbox, status):
        self.corrections_f.write(json.dumps({"id": pid, "bbox": bbox, "status": status}) + "\n")
        self.corrections_f.flush()

    def save_current(self):
        if self.current_box_canvas is None:
            self.status_label.config(text="No box yet -- drag to draw one, or use 'No tablet visible'.")
            return
        pid = str(self.current_item()["id"])
        cx1, cy1, cx2, cy2 = self.current_box_canvas
        bbox = [[cx1 / self.scale, cy1 / self.scale], [cx2 / self.scale, cy2 / self.scale]]
        self._write(pid, bbox, "ok")
        self.advance()

    def mark_bad(self):
        pid = str(self.current_item()["id"])
        self._write(pid, None, "no_tablet")
        self.advance()

    def skip(self):
        self.advance()

    def advance(self):
        self.idx += 1
        self.drag_start = None
        self.load_current()


def main():
    items = load_manifest()
    path_index = build_path_index()
    manifest_ids = {str(it["id"]) for it in items}
    all_done_ids = set()
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    all_done_ids.add(str(json.loads(line)["id"]))
                except Exception:
                    pass
    # Corrections can reference ids no longer in the current manifest (e.g.
    # left over from an earlier collect_vision_dataset.py run whose sample
    # changed) -- only count those still actually in scope, so the progress
    # counter can't show more "done" than there is "total".
    done_ids = all_done_ids & manifest_ids
    stale = len(all_done_ids) - len(done_ids)
    if stale:
        print(f"Note: {stale} past corrections reference ids no longer in the current manifest -- ignored for progress counting.")

    root = tk.Tk()
    ReviewApp(root, items, done_ids, path_index)
    root.mainloop()


if __name__ == "__main__":
    main()
