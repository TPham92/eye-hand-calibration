import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import mm

# ===================== CONFIG =====================

# ChArUco board configuration (maximized for letter in landscape)
BOARD_SQUARES_X = 7
BOARD_SQUARES_Y = 5

# Maximized square size for 7×5 on 11" wide (landscape)
SQUARE_LENGTH_MM = 39.0    # full chessboard square size in mm
MARKER_LENGTH_MM = 31.0    # inner ArUco marker size in mm (~80% of square)

MARGIN_MM = 2.0            # white border around the board in mm

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Render resolution for the raster image embedded in PDF
RENDER_DPI = 300

# Output PDF filename
OUTPUT_PDF = f"charuco_board_{BOARD_SQUARES_X}x{BOARD_SQUARES_Y}_{int(SQUARE_LENGTH_MM)}mm.pdf"


# ===================== HELPERS =====================

def mm_to_pixels(mm_val: float, dpi: float) -> int:
    # 1 inch = 25.4 mm
    return int(round(mm_val * dpi / 25.4))


def main():
    # 1) Create dictionary + ChArUco board (constructor API)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        SQUARE_LENGTH_MM,
        MARKER_LENGTH_MM,
        aruco_dict
    )

    # 2) Physical size of the board (including margin) in mm
    board_w_mm = BOARD_SQUARES_X * SQUARE_LENGTH_MM
    board_h_mm = BOARD_SQUARES_Y * SQUARE_LENGTH_MM

    total_w_mm = board_w_mm + 2 * MARGIN_MM
    total_h_mm = board_h_mm + 2 * MARGIN_MM

    print(f"Board (no margin):    {board_w_mm:.1f} × {board_h_mm:.1f} mm")
    print(f"Board (with margin):  {total_w_mm:.1f} × {total_h_mm:.1f} mm")

    # 3) Render board as a grayscale image at RENDER_DPI
    board_w_px = mm_to_pixels(board_w_mm, RENDER_DPI)
    board_h_px = mm_to_pixels(board_h_mm, RENDER_DPI)

    board_img = board.generateImage((board_w_px, board_h_px))

    # Add margin in pixels around the rendered board
    margin_px = mm_to_pixels(MARGIN_MM, RENDER_DPI)
    total_w_px = board_w_px + 2 * margin_px
    total_h_px = board_h_px + 2 * margin_px

    print(f"Rendered pixels:      {total_w_px} × {total_h_px} px @ {RENDER_DPI} DPI")

    canvas_img = 255 * np.ones((total_h_px, total_w_px), dtype=np.uint8)
    y0 = margin_px
    x0 = margin_px
    canvas_img[y0:y0 + board_h_px, x0:x0 + board_w_px] = board_img

    # Convert to PIL image and save to a temporary PNG
    pil_img = Image.fromarray(canvas_img)
    temp_png = "charuco_board_tmp.png"
    pil_img.save(temp_png)

    # 4) Create a PDF (US Letter, landscape) and place the image at true physical size
    page_w_pt, page_h_pt = landscape(letter)   # in points (1 pt = 1/72 inch)

    # Convert board size from mm to PDF points via reportlab's mm unit
    board_w_pt = total_w_mm * mm
    board_h_pt = total_h_mm * mm

    # Center the board on the page
    x_pt = (page_w_pt - board_w_pt) / 2.0
    y_pt = (page_h_pt - board_h_pt) / 2.0

    c = canvas.Canvas(OUTPUT_PDF, pagesize=landscape(letter))

    # Draw the image with the specified physical size
    c.drawImage(
        temp_png,
        x_pt,
        y_pt,
        width=board_w_pt,
        height=board_h_pt,
        preserveAspectRatio=True,
        mask='auto'
    )

    c.showPage()
    c.save()

    print(f"Saved PDF: {OUTPUT_PDF}")
    print("Print this PDF at 100% / Actual size (no scaling).")
    print("Then measure:")
    print(f"  • One full square ≈ {SQUARE_LENGTH_MM} mm")
    print(f"  • Inner black marker ≈ {MARKER_LENGTH_MM} mm")


if __name__ == "__main__":
    main()
