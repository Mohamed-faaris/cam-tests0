#!/usr/bin/env python3
"""
Generate AprilTags (tag36h11) with multiple layouts per A4 page.
Uses ctypes to access the C library's tag family codes for generation.
"""

import ctypes
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
DPI = 300
MM_TO_PX = DPI / 25.4

A4_WIDTH_PX = int(A4_WIDTH_MM * MM_TO_PX)
A4_HEIGHT_PX = int(A4_HEIGHT_MM * MM_TO_PX)
MARGIN_MM = 10
MARGIN_PX = int(MARGIN_MM * MM_TO_PX)

TAG_FAMILY = "tag36h11"
MAX_TAG_ID = 587


class ApriltagGenerator:
    """Generate AprilTag images using the C library's tag family data."""
    
    def __init__(self, family_name: str = "tag36h11"):
        self.family_name = family_name
        self.lib = self._load_library()
        self._setup_types()
        
        self.family_ptr = self.lib.apriltag_family_create(family_name.encode('ascii'))
        
        if not self.family_ptr:
            raise RuntimeError(f"Failed to create tag family: {family_name}")
        
        # Cast to correct struct type
        self.family = ctypes.cast(self.family_ptr, ctypes.POINTER(_ApriltagFamily)).contents
        self.ncodes = self.family.ncodes
        self.codes = self.family.codes
        self.d = self.family.d  # data bits (6 for tag36h11)
        self.h = self.family.h  # hamming distance
        
        print(f"Loaded {family_name}: {self.ncodes} codes, {self.d}x{self.d} data, h={self.h}")
    
    def _setup_types(self):
        """Define C struct types for the library."""
        global _ApriltagFamily, _ZArray
        
        class _ApriltagFamily(ctypes.Structure):
            _fields_ = [
                ('ncodes', ctypes.c_int32),
                ('codes', ctypes.POINTER(ctypes.c_uint64)),
                ('black_border', ctypes.c_int32),
                ('d', ctypes.c_int32),
                ('h', ctypes.c_int32),
                ('name', ctypes.c_char_p),
            ]
        
        class _ZArray(ctypes.Structure):
            _fields_ = [
                ('el_sz', ctypes.c_size_t),
                ('size', ctypes.c_int),
                ('alloc', ctypes.c_int),
                ('data', ctypes.c_void_p)
            ]
    
    def _load_library(self):
        """Load the apriltag C library."""
        uname0 = os.uname()[0]
        if uname0 == 'Darwin':
            extension = '.dylib'
        else:
            extension = '.so'
        
        filename = f'libapriltag{extension}'
        
        # Try installed location first
        import apriltag
        fpath_installed = os.path.dirname(apriltag.__file__) + "/" + filename
        if os.path.exists(fpath_installed):
            return ctypes.CDLL(fpath_installed)
        
        return ctypes.CDLL(filename)
    
    def get_tag_bits(self, tag_id: int) -> np.ndarray:
        """Get the 6x6 data bits for a tag ID."""
        if tag_id >= self.ncodes:
            raise ValueError(f"Tag ID {tag_id} exceeds maximum {self.ncodes - 1}")
        
        code = self.codes[tag_id]
        bits = np.zeros((self.d, self.d), dtype=np.uint8)
        
        for y in range(self.d):
            for x in range(self.d):
                bit_idx = y * self.d + x
                bits[y, x] = (code >> bit_idx) & 1
        
        return bits
    
    def create_tag_image(self, tag_id: int, size_px: int, border: int = 1) -> np.ndarray:
        """Create a tag image with white border."""
        bits = self.get_tag_bits(tag_id)
        
        # Tag is (d+2*border) x (d+2*border) modules
        modules = self.d + 2 * border
        module_size = size_px // modules
        actual_size = modules * module_size
        
        img = np.zeros((actual_size, actual_size), dtype=np.uint8)
        
        # Fill data bits (white = 1, black = 0 for AprilTag)
        for y in range(self.d):
            for x in range(self.d):
                if bits[y, x]:
                    y_start = (y + border) * module_size
                    y_end = y_start + module_size
                    x_start = (x + border) * module_size
                    x_end = x_start + module_size
                    img[y_start:y_end, x_start:x_end] = 255
        
        return img
    
    def __del__(self):
        if hasattr(self, 'family_ptr') and self.family_ptr:
            self.lib.apriltag_family_destroy(self.family_ptr)


def add_corner_marker_and_info(pil_img: Image.Image, tag_id: int, tag_size_mm: float) -> Image.Image:
    """Add top-right corner marker, ID label, and size info to tag."""
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    
    # Top-right corner marker (red L-shape)
    marker_size = max(10, w // 20)
    draw.rectangle([w - marker_size, 0, w, marker_size], fill="red")
    draw.line([(w - marker_size, 0), (w, marker_size)], fill="red", width=2)
    draw.line([(w, 0), (w - marker_size, marker_size)], fill="red", width=2)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, w // 25))
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, w // 30))
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # ID label top-left
    text = f"ID:{tag_id}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.rectangle([5, 5, 5 + text_w + 4, 5 + text_h + 4], fill="white", outline="black")
    draw.text((7, 7), text, fill="black", font=font)
    
    # Size label bottom-right
    size_text = f"{tag_size_mm:.1f}mm"
    bbox = draw.textbbox((0, 0), size_text, font=font_small)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.rectangle([w - text_w - 9, h - text_h - 9, w - 5, h - 5], fill="white", outline="black")
    draw.text((w - text_w - 7, h - text_h - 7), size_text, fill="black", font=font_small)
    
    return pil_img


def add_scale_bar(page: Image.Image, tag_size_mm: float, tags_per_row: int) -> Image.Image:
    """Add a scale bar at the bottom of the page."""
    draw = ImageDraw.Draw(page)
    w, h = page.size
    
    scale_length_mm = 50
    scale_length_px = int(scale_length_mm * MM_TO_PX)
    
    y_pos = h - 60
    x_start = MARGIN_PX
    x_end = x_start + scale_length_px
    
    draw.line([(x_start, y_pos), (x_end, y_pos)], fill="black", width=2)
    draw.line([(x_start, y_pos - 5), (x_start, y_pos + 5)], fill="black", width=2)
    draw.line([(x_end, y_pos - 5), (x_end, y_pos + 5)], fill="black", width=2)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
    
    draw.text((x_start, y_pos + 10), f"{scale_length_mm} mm", fill="black", font=font)
    draw.text((x_start, y_pos + 30), f"Tag size: {tag_size_mm:.1f} mm", fill="black", font=font)
    draw.text((x_start, y_pos + 50), f"Family: {TAG_FAMILY}", fill="black", font=font)
    
    return page


def calculate_optimal_tag_size(tags_per_page: int) -> tuple:
    """Calculate optimal tag size for given layout on A4."""
    layouts = {
        2: (1, 2),
        4: (2, 2),
        8: (2, 4),
    }
    
    cols, rows = layouts[tags_per_page]
    
    available_w = A4_WIDTH_PX - 2 * MARGIN_PX
    available_h = A4_HEIGHT_PX - 2 * MARGIN_PX
    
    gap_px = int(5 * MM_TO_PX)
    
    max_tag_w = (available_w - (cols - 1) * gap_px) // cols
    max_tag_h = (available_h - (rows - 1) * gap_px) // rows
    
    tag_size_px = min(max_tag_w, max_tag_h)
    tag_size_mm = tag_size_px / MM_TO_PX
    
    return tag_size_px, tag_size_mm, cols, rows, gap_px


def generate_layout(generator: ApriltagGenerator, tags_per_page: int, start_id: int = 0) -> tuple:
    """Generate a page with specified number of tags."""
    tag_size_px, tag_size_mm, cols, rows, gap_px = calculate_optimal_tag_size(tags_per_page)
    
    page = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")
    
    total_w = cols * tag_size_px + (cols - 1) * gap_px
    total_h = rows * tag_size_px + (rows - 1) * gap_px
    start_x = (A4_WIDTH_PX - total_w) // 2
    start_y = (A4_HEIGHT_PX - total_h) // 2
    
    tag_id = start_id
    for row in range(rows):
        for col in range(cols):
            if tag_id >= MAX_TAG_ID:
                tag_id = 0
            
            tag_img = generator.create_tag_image(tag_id, tag_size_px)
            pil_tag = Image.fromarray(tag_img).convert("RGB")
            pil_tag = add_corner_marker_and_info(pil_tag, tag_id, tag_size_mm)
            
            x = start_x + col * (tag_size_px + gap_px)
            y = start_y + row * (tag_size_px + gap_px)
            page.paste(pil_tag, (x, y))
            
            tag_id += 1
    
    page = add_scale_bar(page, tag_size_mm, cols)
    
    # Title
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    title = f"{tags_per_page} AprilTags per page ({TAG_FAMILY})"
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((A4_WIDTH_PX - text_w) // 2, 30), title, fill="black", font=font)
    
    return page, tag_id


def main():
    print(f"A4 page: {A4_WIDTH_PX}x{A4_HEIGHT_PX} px ({A4_WIDTH_MM}x{A4_HEIGHT_MM} mm @ {DPI} DPI)")
    print(f"Tag family: {TAG_FAMILY}")
    print()
    
    generator = ApriltagGenerator(TAG_FAMILY)
    
    for tags_per_page in [2, 4, 8]:
        tag_size_px, tag_size_mm, cols, rows, gap_px = calculate_optimal_tag_size(tags_per_page)
        print(f"{tags_per_page} tags/page: {cols}x{rows} grid, tag={tag_size_px}px ({tag_size_mm:.1f}mm), gap={gap_px}px")
        
        page, next_id = generate_layout(generator, tags_per_page, start_id=tags_per_page * 10)
        
        filename = f"apriltags_{tags_per_page}perpage.png"
        page.save(filename, dpi=(DPI, DPI))
        print(f"  Saved: {filename}")
    
    # Combined image (3 pages stacked)
    print("\nGenerating combined image...")
    combined = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX * 3), "white")
    for i, tags_per_page in enumerate([2, 4, 8]):
        page, _ = generate_layout(generator, tags_per_page, start_id=tags_per_page * 10)
        combined.paste(page, (0, i * A4_HEIGHT_PX))
    
    combined.save("apriltags_all_layouts.png", dpi=(DPI, DPI))
    print("Saved combined: apriltags_all_layouts.png")
    
    # PDF with all layouts
    print("Generating PDF...")
    pdf_pages = []
    for tags_per_page in [2, 4, 8]:
        page, _ = generate_layout(generator, tags_per_page, start_id=tags_per_page * 10)
        pdf_pages.append(page)
    
    pdf_pages[0].save(
        "apriltags_all_layouts.pdf",
        save_all=True,
        append_images=pdf_pages[1:],
        resolution=DPI
    )
    print("Saved PDF: apriltags_all_layouts.pdf")


if __name__ == "__main__":
    main()