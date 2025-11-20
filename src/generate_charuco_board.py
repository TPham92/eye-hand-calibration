import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# ===================== CONFIG LOADING =====================

CONFIG_PATH = Path(__file__).with_name("config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

BOARD_SQUARES_X = config["charuco_marker"]["columns"]             # number of columns
BOARD_SQUARES_Y = config["charuco_marker"]["rows"]                # number of rows
SQUARE_LENGTH_MM = config["charuco_marker"]["square_size"]        # chessboard square size in mm
MARKER_LENGTH_MM = config["charuco_marker"]["marker_size"]        # inner ArUco marker size in mm

MARGIN_MM = config["charuco_marker"]["margin"]                    # white border around the board in mm
RENDER_DPI = config["charuco_marker"]["render_dpi"]               # render resolution for the embedded raster image
ARUCO_DICT_TYPE = getattr(cv2.aruco, config["charuco_marker"]["dictionary"]) # ArUco dictionary type

OUTPUT_PDF = (
    f"charuco_board_{BOARD_SQUARES_X}x{BOARD_SQUARES_Y}_"
    f"{int(SQUARE_LENGTH_MM)}mm.pdf"
)

# ===================== HELPERS =====================

def mm_to_pixels(mm_val: float, dpi: float) -> int:
    # 1 inch = 25.4 mm
    return int(round(mm_val * dpi / 25.4))


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        SQUARE_LENGTH_MM,
        MARKER_LENGTH_MM,
        aruco_dict
    )

    # Physical board size in mm (without margin)
    board_w_mm = BOARD_SQUARES_X * SQUARE_LENGTH_MM
    board_h_mm = BOARD_SQUARES_Y * SQUARE_LENGTH_MM

    total_w_mm = board_w_mm + 2 * MARGIN_MM
    total_h_mm = board_h_mm + 2 * MARGIN_MM

    # Render at chosen DPI
    board_w_px = mm_to_pixels(board_w_mm, RENDER_DPI)
    board_h_px = mm_to_pixels(board_h_mm, RENDER_DPI)

    board_img = board.generateImage((board_w_px, board_h_px))

    # Add white margin around the board
    margin_px = mm_to_pixels(MARGIN_MM, RENDER_DPI)
    total_w_px = board_w_px + 2 * margin_px
    total_h_px = board_h_px + 2 * margin_px

    print(f"Rendered pixels:      {total_w_px} × {total_h_px} px @ {RENDER_DPI} DPI")

    canvas_img = 255 * np.ones((total_h_px, total_w_px), dtype=np.uint8)
    y0 = margin_px
    x0 = margin_px
    canvas_img[y0:y0 + board_h_px, x0:x0 + board_w_px] = board_img

    # Convert numpy → PIL → ImageReader
    pil_img = Image.fromarray(canvas_img)
    img_reader = ImageReader(pil_img)

    # Create PDF (US Letter, landscape)
    page_w_pt, page_h_pt = landscape(letter)

    # Convert physical mm → PDF points
    board_w_pt = total_w_mm * mm
    board_h_pt = total_h_mm * mm

    # Center on page
    x_pt = (page_w_pt - board_w_pt) / 2.0
    y_pt = (page_h_pt - board_h_pt) / 2.0

    c = canvas.Canvas(OUTPUT_PDF, pagesize=landscape(letter))

    # Draw the image at the correct physical size
    c.drawImage(
        img_reader,
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
    print(f"  • One full square ≈ {SQUARE_LENGTH_MM} mm")
    print(f"  • Inner black marker ≈ {MARKER_LENGTH_MM} mm")

if __name__ == "__main__":
    main()
