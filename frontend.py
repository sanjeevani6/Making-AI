
import os
import queue
import threading
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from PIL import Image

import backend
from image_utils import read_image
from table_extractor import (
    cells_to_file, cells_to_grid, extract_tables_from_image,
)
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class OCRApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("DocuScan | OCR Workspace")
        self.geometry("1250x800")
        self.minsize(1000, 650)
        self.configure(fg_color="#F4F6FA")

        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.current_text = ""
        self.current_mode = "Printed OCR"
        self.busy = False
        self.table_cells = []
        self.result_queue = queue.Queue()
        self.mode_buttons = {}
        self.pdf_font = self._register_pdf_font()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_workspace()
        self._set_status("Ready to upload a document")
        self._highlight_mode()

    @staticmethod
    def _register_pdf_font():
        """Use a Unicode TrueType font in PDFs when one is available."""
        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont("DocuFont", path))
                    return "DocuFont"
                except Exception:
                    pass
        return "Helvetica"

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self, width=220, corner_radius=0,
            fg_color="#182235"
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        ctk.CTkLabel(
            self.sidebar,
            text="▣  DocuScan",
            font=("Segoe UI", 25, "bold"),
            text_color="white"
        ).pack(padx=20, pady=(28, 6), anchor="w")

        ctk.CTkLabel(
            self.sidebar,
            text="DOCUMENT WORKSPACE",
            font=("Segoe UI", 10, "bold"),
            text_color="#9BAAC2"
        ).pack(padx=22, pady=(0, 28), anchor="w")

        self._nav_button("＋  Upload Document", self.upload_image)
        self.mode_buttons["Printed OCR"] = self._nav_button(
            "▤  Printed Text OCR",
            lambda: self.select_mode("Printed OCR"))
        self.mode_buttons["Handwriting OCR"] = self._nav_button(
            "✎  Handwriting OCR",
            lambda: self.select_mode("Handwriting OCR"))
        self.mode_buttons["Table Extraction"] = self._nav_button(
            "▦  Table Extraction",
            lambda: self.select_mode("Table Extraction"))

        ctk.CTkFrame(
            self.sidebar, height=1, fg_color="#344158"
        ).pack(fill="x", padx=18, pady=22)

        ctk.CTkLabel(
            self.sidebar,
            text="EXPORT RESULTS",
            font=("Segoe UI", 10, "bold"),
            text_color="#9BAAC2"
        ).pack(padx=22, pady=(0, 10), anchor="w")

        self._nav_button("↓  Save Text", self.save_text)
        self._nav_button("▧  Export PDF", self.save_pdf)
        self._nav_button("▦  Export Table (CSV/XLSX)", self.save_table)
        self._nav_button("▣  Save Processed Image", self.save_image)

        ctk.CTkLabel(
            self.sidebar,
            text="OCR Document Scanner\nTesseract · TrOCR",
            text_color="#9BAAC2",
            font=("Segoe UI", 11),
            justify="left"
        ).pack(side="bottom", padx=20, pady=24, anchor="w")

    def _nav_button(self, text, command):
        button = ctk.CTkButton(
            self.sidebar,
            text=text,
            command=command,
            height=42,
            corner_radius=9,
            anchor="w",
            fg_color="transparent",
            hover_color="#2A3850",
            text_color="#E8EDF5",
            font=("Segoe UI", 13)
        )
        button.pack(fill="x", padx=12, pady=3)
        return button

    def _highlight_mode(self):
        for mode, button in self.mode_buttons.items():
            active = mode == self.current_mode
            button.configure(fg_color="#2F4B7C" if active else "transparent")

    def _build_workspace(self):
        self.workspace = ctk.CTkFrame(
            self, corner_radius=0, fg_color="#F4F6FA"
        )
        self.workspace.grid(
            row=0, column=1, sticky="nsew", padx=24, pady=20
        )
        self.workspace.grid_columnconfigure((0, 1), weight=1)
        self.workspace.grid_rowconfigure(2, weight=3)
        self.workspace.grid_rowconfigure(4, weight=2)

        # Header
        header = ctk.CTkFrame(
            self.workspace, fg_color="transparent"
        )
        header.grid(row=0, column=0, columnspan=2,
                    sticky="ew", pady=(0, 16))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="OCR Workspace",
            font=("Segoe UI", 27, "bold"),
            text_color="#182235"
        ).grid(row=0, column=0, sticky="w")

        self.mode_label = ctk.CTkLabel(
            header, text="Mode: Printed OCR",
            font=("Segoe UI", 12),
            text_color="#53627A"
        )
        self.mode_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.process_btn = ctk.CTkButton(
            header, text="▶  Process Document",
            command=self.process_document,
            width=190, height=42, corner_radius=10
        )
        self.process_btn.grid(row=0, column=1, rowspan=2, padx=8)

        self.upload_btn = ctk.CTkButton(
            header, text="＋ Upload",
            command=self.upload_image,
            width=110, height=42, corner_radius=10,
            fg_color="#FFFFFF", hover_color="#E8EDF5",
            text_color="#182235",
            border_width=1, border_color="#D8DFEA"
        )
        self.upload_btn.grid(row=0, column=2, rowspan=2)

        # Status bar
        self.status_label = ctk.CTkLabel(
            self.workspace, text="Ready",
            anchor="w", height=30,
            fg_color="#E7ECF5", corner_radius=8,
            text_color="#465570"
        )
        self.status_label.grid(
            row=1, column=0, columnspan=2,
            sticky="ew", pady=(0, 14)
        )

        # Image panels
        self.original_panel = self._image_panel(
            self.workspace, "Original Document"
        )
        self.original_panel.grid(
            row=2, column=0, sticky="nsew", padx=(0, 8), pady=4
        )

        self.processed_panel = self._image_panel(
            self.workspace, "Processed Document"
        )
        self.processed_panel.grid(
            row=2, column=1, sticky="nsew", padx=(8, 0), pady=4
        )

        # Results header
        result_header = ctk.CTkFrame(
            self.workspace, fg_color="transparent"
        )
        result_header.grid(
            row=3, column=0, columnspan=2,
            sticky="ew", pady=(15, 6)
        )
        result_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            result_header, text="Recognized Text",
            font=("Segoe UI", 18, "bold"),
            text_color="#182235"
        ).grid(row=0, column=0, sticky="w")

        self.copy_btn = ctk.CTkButton(
            result_header, text="Copy",
            width=75, height=30,
            command=self.copy_text
        )
        self.copy_btn.grid(row=0, column=1, padx=4)

        self.clear_btn = ctk.CTkButton(
            result_header, text="Clear",
            width=75, height=30,
            command=self.clear_text,
            fg_color="#FFFFFF", hover_color="#E8EDF5",
            text_color="#182235",
            border_width=1, border_color="#D8DFEA"
        )
        self.clear_btn.grid(row=0, column=2, padx=(4, 0))

        self.text_box = ctk.CTkTextbox(
            self.workspace,
            corner_radius=10,
            border_width=1,
            border_color="#D8DFEA",
            fg_color="#FFFFFF",
            text_color="#182235",
            font=("Consolas", 13),
            wrap="word"
        )
        self.text_box.grid(
            row=4, column=0, columnspan=2,
            sticky="nsew", pady=(0, 8)
        )

    def _image_panel(self, parent, title):
        panel = ctk.CTkFrame(
            parent, corner_radius=12,
            fg_color="#FFFFFF",
            border_width=1, border_color="#E0E5EE"
        )
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            panel, text=title,
            font=("Segoe UI", 14, "bold"),
            text_color="#344158"
        ).grid(row=0, column=0, padx=14, pady=12, sticky="w")

        image_label = ctk.CTkLabel(
            panel,
            text="Upload an image to preview it",
            font=("Segoe UI", 13),
            text_color="#8A96A9",
            fg_color="#F7F9FC",
            corner_radius=8
        )
        image_label.grid(
            row=1, column=0, sticky="nsew",
            padx=12, pady=(0, 12)
        )
        panel.image_label = image_label
        return panel

    def _set_status(self, message):
        self.status_label.configure(text=f"  {message}")
        self.update_idletasks()

    def select_mode(self, mode):
        self.current_mode = mode
        self.mode_label.configure(text=f"Mode: {mode}")
        self._highlight_mode()
        self._set_status(f"Selected {mode}")

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select a document image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        image = read_image(path)
        if image is None:
            messagebox.showerror(
                "Invalid image", "The selected file could not be opened."
            )
            return

        self.image_path = path
        self.original_image = image
        self.processed_image = None
        self.table_cells = []
        self._show_cv_image(image, self.original_panel.image_label)
        self.processed_panel.image_label.configure(
            image=None, text="Process the document to see the result."
        )
        self.clear_text()
        self._set_status(f"Loaded: {os.path.basename(path)}")

    def _show_cv_image(self, image, label):
        if image is None:
            label.configure(image=None, text="No image available.")
            label.image = None
            return

        if len(image.shape) == 2:
            # Grayscale images, such as the handwriting preprocessing preview.
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # OpenCV color images use BGR; PIL expects RGB.
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((560, 320))
        photo = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=pil_image.size
        )
        label.configure(image=photo, text="")
        label.image = photo

 
    # ------------------------------------------------------------------
    # Processing runs in a worker thread so the window never freezes.
    # The worker never touches widgets; it reports through a queue.
    # ------------------------------------------------------------------
    def process_document(self):
        if self.busy:
            return

        if not self.image_path:
            messagebox.showinfo(
                "Upload document",
                "Please upload an image first."
            )
            return

        self.busy = True
        self.process_btn.configure(state="disabled")
        first_use = " (the first run loads the model, please wait)" \
            if self.current_mode == "Handwriting OCR" else ""
        self._set_status(f"Processing with {self.current_mode}...{first_use}")

        threading.Thread(
            target=self._worker,
            args=(self.current_mode, self.image_path),
            daemon=True
        ).start()
        self.after(100, self._poll_worker)

    def _worker(self, mode, path):
        try:
            if mode == "Printed OCR":
                result = backend.handle_user_choice(path, "printed")
                if result["processed"] is None:
                    raise ValueError("Could not preprocess this image.")
                info = result.get("info", {})
                result["status"] = (
                    f"Printed OCR completed "
                    f"({info.get('engine', '?')}, "
                    f"confidence {info.get('confidence', 0):.0f}%)."
                )

            elif mode == "Handwriting OCR":
                def progress(done, total, message):
                    self.result_queue.put(
                        ("progress", f"{message}: {done}/{total} lines"))
                result = backend.handle_user_choice(
                    path, "handwritten", progress_cb=progress)
                result["status"] = "Handwriting recognition completed."

            elif mode == "Table Extraction":
                image, cells = extract_tables_from_image(path)
                result = {"processed": image, "cells": cells,
                          "text": "", "status": ""}
                if cells:
                    grid = cells_to_grid(cells)
                    result["text"] = "\n".join("\t".join(r) for r in grid)
                    result["status"] = (
                        f"Table extraction completed: {len(grid)} rows x "
                        f"{len(grid[0])} columns.")
                else:
                    result["status"] = "No table cells detected."
            else:
                raise ValueError(f"Unsupported processing mode: {mode}")

            self.result_queue.put(("done", mode, result))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def _poll_worker(self):
        finished = False
        try:
            while True:
                message = self.result_queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    self._set_status(message[1])
                elif kind == "done":
                    self._apply_result(message[1], message[2])
                    finished = True
                elif kind == "error":
                    self._set_status("Processing failed.")
                    messagebox.showerror(
                        "Processing error",
                        f"An error occurred:\n{message[1]}"
                    )
                    finished = True
        except queue.Empty:
            pass

        if finished:
            self.busy = False
            self.process_btn.configure(state="normal")
        else:
            self.after(100, self._poll_worker)

    def _apply_result(self, mode, result):
        processed = result.get("processed")
        self.processed_image = processed
        if processed is not None:
            self._show_cv_image(processed, self.processed_panel.image_label)
        else:
            self.processed_panel.image_label.configure(
                image=None, text="Could not create the processed preview."
            )
            self.processed_panel.image_label.image = None

        self._set_text(result.get("text", ""))
        self._set_status(result.get("status", "Done."))

        if mode == "Table Extraction":
            self.table_cells = result.get("cells", [])
            if not self.table_cells:
                messagebox.showinfo(
                    "No tables found",
                    "No table cells were detected in this image."
                )
            else:
                self._set_status(
                    result["status"] + "  Use 'Export Table' to save it.")

    def _set_text(self, text):
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text or "")
        self.current_text = text or ""

    def clear_text(self):
        if hasattr(self, "text_box"):
            self.text_box.delete("1.0", "end")
        self.current_text = ""

    def copy_text(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Nothing to copy", "There is no text to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self._set_status("Text copied to clipboard.")

    def save_text(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("No text", "Process a document or enter text first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
            self._set_status(f"Text saved: {os.path.basename(path)}")

    def save_image(self):
        if self.processed_image is None:
            messagebox.showinfo(
                "No processed image",
                "Process a document first."
            )
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")]
        )
        if path:
            ext = os.path.splitext(path)[1] or ".png"
            ok, buffer = cv2.imencode(ext, self.processed_image)
            if ok:
                buffer.tofile(path)  # works with non-ASCII paths on Windows
                self._set_status(f"Image saved: {os.path.basename(path)}")
            else:
                messagebox.showerror("Save failed", "Could not save the image.")

    def save_table(self):
        if not self.table_cells:
            messagebox.showinfo(
                "No table",
                "Run Table Extraction on an image with a table first."
            )
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("CSV files", "*.csv")]
        )
        if not path:
            return
        try:
            cells_to_file(self.table_cells, path)
            self._set_status(f"Table saved: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def save_pdf(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("No text", "Process a document or enter text first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not path:
            return

        pdf = canvas.Canvas(path, pagesize=letter)
        page_width, page_height = letter
        y = page_height - 50
        pdf.setFont(self.pdf_font, 11)

        for line in text.splitlines() or [""]:
            wrapped = simpleSplit(line, self.pdf_font, 11, page_width - 60)
            for wrapped_line in wrapped or [""]:
                if y < 50:
                    pdf.showPage()
                    pdf.setFont(self.pdf_font, 11)
                    y = page_height - 50
                pdf.drawString(30, y, wrapped_line)
                y -= 15

        pdf.save()
        self._set_status(f"PDF saved: {os.path.basename(path)}")


if __name__ == "__main__":
    app = OCRApp()
    app.mainloop()
